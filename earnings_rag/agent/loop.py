import json
from dataclasses import dataclass, field

from earnings_rag.agent.tools import Tool, DEFAULT_TOOLS
from earnings_rag.config import settings

SYSTEM_PROMPT = """You are a financial research assistant with tools for searching SEC 10-K filings.

Scope: the filings cover only NVIDIA, Apple, and Capital One. If the question is about any other company, say "The provided filings do not address this." without searching.

Rules:
- Use the tools to find evidence before answering. Search again with a different query or company if the first results don't cover the question.
- Answer ONLY from tool results. Cite each claim inline with the passage id in square brackets, like [NVDA_2025-01-26_0005], or the fact id for a figure, like [NVDA_revenue_FY2025].
- For numbers (revenue, income, EPS, cash flow), use get_financials; use search_filings for explanations, strategy and risks. A question may need both.
- Never do arithmetic yourself. Use the calculate tool for every derived number (growth, margins, differences, ratios), and use only numbers that appear in tool results, passing get_financials values to calculate exactly as returned.
- Use only passages that directly address the question. Ignore retrieved passages on other topics, even from the right company. Do not pad the answer to a fixed number of points.
- If any passage discusses the subject of the question, answer from it, even if the information is partial, hedged, or framed as a risk.
- Only say "The provided filings do not address this." if no passage discusses the subject of the question."""


@dataclass
class AgentResult:
    answer: str
    trace: list[dict] = field(default_factory=list)  # one entry per tool call: step, tool, arguments, result, new_results
    usage: list[dict] = field(default_factory=list)  # one entry per model call: call, input, output, finish_reason
    steps: int = 0                                   # model calls used, not counting the forced final one
    truncated: bool = False                          # True if the step budget ran out and the answer was forced
    cut_off: bool = False                            # True if the final answer hit max_tokens mid-answer

    @property
    def input_tokens(self) -> int:
        return sum(u['input'] for u in self.usage)

    @property
    def output_tokens(self) -> int:
        return sum(u['output'] for u in self.usage)


def _assistant_message(msg) -> dict:
    """Convert the SDK's message object to the plain dict the API wants sent back."""
    return {
        'role': 'assistant',
        'content': msg.content,
        'tool_calls': [
            {'id': tc.id, 'type': 'function',
             'function': {'name': tc.function.name, 'arguments': tc.function.arguments}}
            for tc in msg.tool_calls
        ],
    }


def annotate_novelty(result: str, seen_ids: set[str]) -> tuple[str, int | None]:
    """
    For search-style results ({"results": [{"id": ...}, ...]}), count the passages not returned by an earlier
    call and, if some were repeats, add a note for the model. Advisory only: the model may still search again.

    Returns (result, new_count). new_count is None for errors and any other result shape, which pass through.
    """
    try:
        data = json.loads(result)
        ids = [r['id'] for r in data['results']]
    except (ValueError, KeyError, TypeError):
        return result, None

    new = [i for i in ids if i not in seen_ids]
    seen_ids.update(ids)

    if len(new) < len(ids):
        data['note'] = (f'{len(ids) - len(new)} of {len(ids)} passages were already returned by earlier calls. '
                        'Try a different query or company, or answer from what you have.')
        return json.dumps(data), len(new)
    return result, len(new)


def run_agent(question: str, client, tools: list[Tool] | None = None,
              max_steps: int | None = None, model: str | None = None) -> AgentResult:
    """
    The whole agent: call the model; if it asks for tools, run them, append the results, and call it again.
    Stops when the model answers without asking for a tool, or when the step budget is spent.

    `client` is anything with the OpenAI `chat.completions.create` interface, so tests can pass a scripted fake.
    """
    tools = tools if tools is not None else DEFAULT_TOOLS
    max_steps = max_steps if max_steps is not None else settings.agent_max_steps
    model = model or settings.llm_model

    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]
    trace = []
    usage = []
    seen_ids: set[str] = set()

    def call_model(**extra):
        response = client.chat.completions.create(
            model=model, max_tokens=settings.agent_max_tokens, temperature=0,
            messages=messages, tools=schemas, **extra,
        )
        choice = response.choices[0]
        usage.append({'call': len(usage) + 1, 'input': response.usage.prompt_tokens,
                      'output': response.usage.completion_tokens, 'finish_reason': choice.finish_reason})
        return choice

    for step in range(1, max_steps + 1):
        choice = call_model()
        msg = choice.message

        if not msg.tool_calls:
            return AgentResult(answer=msg.content or '', trace=trace, usage=usage, steps=step,
                               cut_off=choice.finish_reason == 'length')

        messages.append(_assistant_message(msg))
        for tc in msg.tool_calls:
            tool = by_name.get(tc.function.name)
            if tool is None:
                result = f'{{"error": "Unknown tool {tc.function.name}"}}'
            else:
                result = tool.call(tc.function.arguments)
            result, new_results = annotate_novelty(result, seen_ids)

            trace.append({'step': step, 'tool': tc.function.name, 'arguments': tc.function.arguments,
                          'result': result, 'new_results': new_results})
            messages.append({'role': 'tool', 'tool_call_id': tc.id, 'content': result})

    # Budget spent. A partial answer from what it has gathered beats an error, so forbid further tool calls.
    choice = call_model(tool_choice='none')
    return AgentResult(answer=choice.message.content or '', trace=trace, usage=usage, steps=max_steps,
                       truncated=True, cut_off=choice.finish_reason == 'length')
