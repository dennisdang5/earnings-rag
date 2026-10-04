import json
from dataclasses import dataclass, field
from datetime import date

from earnings_rag.agent.citations import check_citations, check_missing_filings, revision_request
from earnings_rag.agent.tools import Tool, DEFAULT_TOOLS
from earnings_rag.config import settings
from earnings_rag.periods import period_note, resolve_periods
from earnings_rag.store import text_periods

SYSTEM_PROMPT = """You are a financial research assistant with tools for searching SEC 10-K (annual) and 10-Q (quarterly) filings.

Scope: the filings cover only NVIDIA, Apple, and Capital One. If the question is about any other company, say "The provided filings do not address this." without searching.

Rules:
- search_filings searches annual reports by default. For a quarter, a quarterly report, or the latest or most recent report, set period (Q1-Q3) or latest=true, and say which filing a statement comes from, for example "NVIDIA's 10-Q for Q2 FY2027". Never describe annual-report text as a quarterly report.
- If search_filings finds nothing for the period or filing the user asked for, begin the answer by saying that filing is not available and which filings exist. Do not present another period's passages as the answer to the period asked about.
- Use the tools to find evidence before answering. Search again with a different query or company if the first results don't cover the question.
- Except for the scope rule above, always search before answering: never say the filings do not address a question without having searched.
- Answer ONLY from tool results. Cite each claim inline with the passage id in square brackets, like [NVDA_2025-01-26_0005], or the fact id for a figure, like [NVDA_revenue_FY2025].
- For numbers (revenue, income, EPS, cash flow), use get_financials; use search_filings for explanations, strategy and risks. A question may need both.
- Every number in the answer needs a citation right after it: the fact id if it came from get_financials, the calc id (like [calc_1]) if it came from calculate, the passage id if a passage states it. A computed number (growth, a margin, a difference) cites its calc id, never the facts it was computed from. If you report a number from get_financials, cite that fact id even when a passage also mentions it.
- If get_financials says a company does not report a metric, say exactly that and, if relevant, give the related metrics it does report. Do not use the "do not address" sentence for this.
- Never do arithmetic yourself. Use the calculate tool for every derived number (growth, margins, differences, ratios), and use only numbers that appear in tool results, passing get_financials values to calculate exactly as returned.
- Use only passages that directly address the question. Ignore retrieved passages on other topics, even from the right company. Do not pad the answer to a fixed number of points.
- If any passage discusses the subject of the question, answer from it, even if the information is partial, hedged, or framed as a risk.
- Only say "The provided filings do not address this." if no passage discusses the subject of the question."""



def date_context(today: date, filings: list[dict]) -> str:
    """
    Today's date and each company's newest 10-K and 10-Q, appended to the system prompt. Without them the model
    resolved "next fiscal year" from its training data (NVIDIA FY2024) and answered from that old filing (8 of 8 runs).
    The period end dates show each fiscal calendar (NVIDIA's year ends in January). `filings` is text_periods() rows.
    """
    newest: dict[str, dict[str, dict]] = {}
    for r in filings:                                  # oldest first, so the last row per form is the newest
        newest.setdefault(r['ticker'], {})[r['form']] = r
    parts = []
    for ticker, forms in newest.items():
        latest = []
        for form, r in sorted(forms.items()):
            quarter = '' if r['fiscal_period'] == 'FY' else f"{r['fiscal_period']} "
            latest.append(f"{form} {quarter}FY{r['fiscal_year']} (ended {r['period']})")
        parts.append(f"{ticker} {', '.join(latest)}")
    return (f"Today is {today.isoformat()}. Newest filings with text: {'; '.join(parts)}. A fiscal year is named for "
            "the calendar year it ends in. Work out relative dates (\"next fiscal year\", \"last quarter\", \"a year "
            "ago\") from today's date and that company's fiscal calendar before searching. A period after a company's "
            "newest filing has not been filed yet.")


@dataclass
class AgentResult:
    answer: str
    trace: list[dict] = field(default_factory=list)  # one entry per tool call: step, tool, arguments, result, new_results
    usage: list[dict] = field(default_factory=list)  # one entry per model call: call, input, output, finish_reason
    steps: int = 0                                   # model calls used, not counting the forced final one
    truncated: bool = False                          # True if the step budget ran out and the answer was forced
    cut_off: bool = False                            # True if the final answer hit max_tokens mid-answer
    revised: bool = False                            # True if the answer checks sent the answer back once
    uncited: list[str] = field(default_factory=list)       # sentences in the final answer with a figure and no [id]
    unknown_ids: list[str] = field(default_factory=list)   # cited ids in the final answer that no tool returned
    unsupported: list[tuple] = field(default_factory=list)  # (figure, sentence): no fact or calc cited in it states it
    unacknowledged_missing: list[str] = field(default_factory=list)  # requested filings found missing, answer silent
    resolved_periods: list[dict] = field(default_factory=list)       # relative periods resolved in code (periods.py)

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


def record_numbers(result: str, values: dict[str, list[float]], computed: dict[str, float]) -> str:
    """
    Give a calculate result an id the answer can cite (calc_1, calc_2, ... per run) and remember its value; remember
    every fact's value (and as_reported) by id. The citation check uses both to see whether a cited id states the
    figure next to it. Other results pass through unchanged.
    """
    try:
        data = json.loads(result)
    except ValueError:
        return result
    if not isinstance(data, dict):
        return result
    if 'expression' in data and isinstance(data.get('result'), (int, float)):
        calc_id = f'calc_{len(computed) + 1}'
        computed[calc_id] = data['result']
        return json.dumps({'id': calc_id, **data})
    for r in data.get('results') or []:
        if isinstance(r, dict) and isinstance(r.get('value'), (int, float)):
            values[r['id']] = [r['value']] + ([r['as_reported']] if 'as_reported' in r else [])
    return result


def missing_filing(result: str) -> dict | None:
    """The filing a search_filings error reports missing ({"label", "fiscal_year"}), or None for any other result."""
    try:
        return json.loads(result).get('missing_filing')
    except (ValueError, AttributeError):
        return None


def run_agent(question: str, client, tools: list[Tool] | None = None,
              max_steps: int | None = None, model: str | None = None, context: str | None = None,
              filings: list[dict] | None = None, today: date | None = None) -> AgentResult:
    """
    The whole agent: call the model; if it asks for tools, run them, append the results, and call it again.
    Stops when the model answers without asking for a tool, or when the step budget is spent.

    `client` is anything with the OpenAI `chat.completions.create` interface, so tests can pass a scripted fake.
    `context` is appended to the system prompt; None builds date_context, '' adds nothing. Relative periods in the
    question are resolved in code and noted beside it (periods.py). `filings` (text_periods() rows, default: the
    database) and `today` are parameters so tests can fix them.
    """
    filings = text_periods() if filings is None else filings
    today = today or date.today()
    if context is None:
        context = date_context(today, filings)
    resolved = resolve_periods(question, filings, today)
    # the model sees the question with the note; the missing-filing check counts the resolved years as asked for
    asked = question + ('\n\n' + period_note(resolved, today) if resolved else '')
    tools = tools if tools is not None else DEFAULT_TOOLS
    max_steps = max_steps if max_steps is not None else settings.agent_max_steps
    model = model or settings.llm_model

    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT + ('\n\n' + context if context else '')},
        {'role': 'user', 'content': asked},
    ]
    trace = []
    usage = []
    seen_ids: set[str] = set()
    missing: list[dict] = []
    values: dict[str, list[float]] = {}   # fact id -> the numbers it states
    computed: dict[str, float] = {}       # calc id -> result
    revised = False

    def check(answer: str) -> dict:
        return {**check_citations(answer, seen_ids, values, computed),
                'unacknowledged_missing': check_missing_filings(answer, asked, missing)}

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
            problems = check(msg.content or '')
            if any(problems.values()) and not revised:
                # one revision turn for every problem, tools still allowed so it can fetch a missing fact; a second
                # failure is accepted
                revised = True
                messages.append({'role': 'assistant', 'content': msg.content})
                messages.append({'role': 'user', 'content': revision_request(problems)})
                continue
            return AgentResult(answer=msg.content or '', trace=trace, usage=usage, steps=step,
                               cut_off=choice.finish_reason == 'length', revised=revised, resolved_periods=resolved, **problems)

        messages.append(_assistant_message(msg))
        for tc in msg.tool_calls:
            tool = by_name.get(tc.function.name)
            if tool is None:
                result = f'{{"error": "Unknown tool {tc.function.name}"}}'
            else:
                result = tool.call(tc.function.arguments)
            result = record_numbers(result, values, computed)
            seen_ids.update(computed)
            result, new_results = annotate_novelty(result, seen_ids)
            if (m := missing_filing(result)) is not None:
                missing.append(m)

            trace.append({'step': step, 'tool': tc.function.name, 'arguments': tc.function.arguments,
                          'result': result, 'new_results': new_results})
            messages.append({'role': 'tool', 'tool_call_id': tc.id, 'content': result})

    # Budget spent. A partial answer from what it has gathered beats an error, so forbid further tool calls.
    choice = call_model(tool_choice='none')
    answer = choice.message.content or ''
    return AgentResult(answer=answer, trace=trace, usage=usage, steps=max_steps, truncated=True,
                       cut_off=choice.finish_reason == 'length', revised=revised, resolved_periods=resolved, **check(answer))
