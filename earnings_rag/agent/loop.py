from dataclasses import dataclass, field

from earnings_rag.agent.tools import Tool, DEFAULT_TOOLS
from earnings_rag.config import settings

SYSTEM_PROMPT = """You are a financial research assistant with tools for searching SEC 10-K filings.

Rules:
- Use the tools to find evidence before answering. Search again with a different query or company if the first results don't cover the question.
- Answer ONLY from tool results. Cite each claim inline with the passage id in square brackets, like [NVDA_2025-01-26_0005].
- If any passage discusses the subject of the question, answer from it, even if the information is partial, hedged, or framed as a risk.
- Only say "The provided filings do not address this." if no passage discusses the subject of the question."""


@dataclass
class AgentResult:
    answer: str
    trace: list[dict] = field(default_factory=list)  # one entry per tool call: step, tool, arguments, result
    steps: int = 0                                   # model calls used, not counting the forced final one
    truncated: bool = False                          # True if the step budget ran out and the answer was forced


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

    def call_model(**extra):
        return client.chat.completions.create(
            model=model, max_tokens=settings.llm_max_tokens, temperature=0,
            messages=messages, tools=schemas, **extra,
        ).choices[0].message

    for step in range(1, max_steps + 1):
        msg = call_model()

        if not msg.tool_calls:
            return AgentResult(answer=msg.content or '', trace=trace, steps=step)

        messages.append(_assistant_message(msg))
        for tc in msg.tool_calls:
            tool = by_name.get(tc.function.name)
            if tool is None:
                result = f'{{"error": "Unknown tool {tc.function.name}"}}'
            else:
                result = tool.call(tc.function.arguments)

            trace.append({'step': step, 'tool': tc.function.name,
                          'arguments': tc.function.arguments, 'result': result})
            messages.append({'role': 'tool', 'tool_call_id': tc.id, 'content': result})

    # Budget spent. A partial answer from what it has gathered beats an error, so forbid further tool calls.
    msg = call_model(tool_choice='none')
    return AgentResult(answer=msg.content or '', trace=trace, steps=max_steps, truncated=True)
