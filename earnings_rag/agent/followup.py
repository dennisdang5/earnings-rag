"""
Follow-up questions: turn "And Apple?" into a question that stands alone, so the agent (which is stateless) can answer it.

The history is only used to phrase the question, never as evidence: the agent fetches every number again and the
citation checks accept only ids returned in this run. So a wrong or made-up history can mislead the phrasing, but not
smuggle in a figure. Any failure returns the question unchanged.
"""
import json
import re
from dataclasses import dataclass

import structlog

from earnings_rag.agent.citations import years_in
from earnings_rag.config import settings
from earnings_rag.periods import PHRASE

log = structlog.get_logger()

MAX_QUESTION_CHARS = 1000  # the same cap /agent/stream puts on a question

REWRITE_PROMPT = """You rewrite a user's follow-up question so it can be understood without the earlier conversation.

Rules:
- A complete question stands alone, even a broad one ("What did Apple report last year?", "How do export controls affect NVIDIA?"); return it word for word. Do not narrow it to the earlier topic.
- A fragment does not stand alone: "And Apple?", "And Apple last quarter?", "What about 2024?", "Why did it grow?". It names a company or period but asks nothing by itself, so add the topic (metric, risk, product) from the earlier conversation.
- Otherwise fill in what it leaves out (the company, the metric, the period) from the earlier conversation.
- Keep period wording exactly as the user wrote it ("last year", "last quarter", "two years ago", "next fiscal year"); do not turn it into a year or a date.
- If the question names a company, keep that company; never replace it with one from the earlier conversation.
- Do not answer the question and do not add facts or figures.
Reply with JSON only: {"question": "<the standalone question>"}"""


@dataclass
class Rewrite:
    question: str   # what the agent should be asked
    original: str   # what the user typed
    changed: bool
    input_tokens: int = 0    # the rewrite call's own cost (0 when no call was made)
    output_tokens: int = 0


def _same(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r'\W+', ' ', s).strip().lower()
    return norm(a) == norm(b)


def keeps_period_wording(original: str, new: str) -> bool:
    """Relative periods ("last year", "last quarter") are resolved in code from each company's calendar (periods.py). A
    rewrite that drops one, or swaps it for a year it took from the history, would skip that. Measured: with history,
    "What did Apple report last year?" came back as "...in fiscal 2024" 4/4 although the prompt says not to."""
    phrases = [m.group(0).lower() for m in PHRASE.finditer(original)]
    return all(p in new.lower() for p in phrases) and (not phrases or years_in(new) <= years_in(original))


def format_history(history: list[dict]) -> str:
    """The last `followup_turns` turns, each answer cut to `followup_answer_chars` (its [ids] name company and year)."""
    turns = history[-settings.followup_turns:]
    return '\n\n'.join(f'User: {t["question"]}\nAssistant: {t["answer"][:settings.followup_answer_chars]}' for t in turns)


def rewrite(question: str, history: list[dict], client, model: str | None = None) -> Rewrite:
    unchanged = Rewrite(question, question, False)
    if not history:
        return unchanged
    spent = (0, 0)
    try:
        resp = client.chat.completions.create(
            model=model or settings.llm_model, temperature=0, max_tokens=150,
            response_format={'type': 'json_object'},
            messages=[{'role': 'system', 'content': REWRITE_PROMPT},
                      {'role': 'user', 'content': f'Earlier conversation:\n{format_history(history)}\n\n'
                                                  f'Follow-up question: {question}'}])
        spent = (resp.usage.prompt_tokens, resp.usage.completion_tokens)
        new = json.loads(resp.choices[0].message.content)['question'].strip()
    except Exception:   # a failed rewrite must never lose the answer
        log.warning('followup_rewrite_failed', exc_info=True)
        return Rewrite(question, question, False, *spent)
    if not new or len(new) > MAX_QUESTION_CHARS or _same(new, question) or not keeps_period_wording(question, new):
        return Rewrite(question, question, False, *spent)
    return Rewrite(new, question, True, *spent)
