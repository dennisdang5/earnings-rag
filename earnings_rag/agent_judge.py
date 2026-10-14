"""
LLM judge for stored agent runs (see DECISIONS.md, agent eval). Groundedness: one call per answer; the answer is split
into numbered claims by code and the judge sees every passage, fact and calculation the run retrieved. Relevance: a
separate call that sees only the question and the answer. The judge never runs in CI; tests use a scripted client.
"""
import json
import re

from earnings_rag.agent.citations import _cited_ids, _sentences
from earnings_rag.config import settings

PROMPT_VERSION = 'v3'
VERDICTS = ('supported', 'partial', 'unsupported', 'not_a_claim')
SUPPORT_LABELS = VERDICTS
RELEVANCE = ('answers', 'partial', 'off_topic')


# A period after these does not end a sentence. "The U.S. government ..." was cut into "The U.S." and "government ...",
# and the fragment was labelled by hand as not a claim (validation sheet L20, L22). Two sentences merged by mistake are
# harmless to a judge; a fragment is not.
ABBREVIATION = re.compile(r'\b(?:U\.S|U\.K|e\.g|i\.e|Inc|Corp|Co|Ltd|No|vs|Mr|Mrs|Dr)\.')
DOT = '․'    # stands in for the period while sentences are split


def split_claims(answer: str) -> list[dict]:
    """
    Numbered sentences. Drafts tend to cite once at the end of a paragraph, so each claim keeps the ids cited in its own
    sentence (`ids`) and in its paragraph (`paragraph_ids`); citation correctness is judged against the second.
    """
    claims = []
    for paragraph in re.split(r'\n\s*\n', ABBREVIATION.sub(lambda m: m.group(0)[:-1] + DOT, answer)):
        paragraph_ids = _cited_ids(paragraph)
        for sentence in (s.replace(DOT, '.') for s in _sentences(paragraph)):
            claims.append({'n': len(claims) + 1, 'text': sentence, 'ids': _cited_ids(sentence),
                           'paragraph_ids': paragraph_ids})
    return claims


def evidence_items(trace: list[dict]) -> dict[str, str]:
    """
    id -> text for what the run's tools returned as evidence: passages, fact rows and calculations. Tool errors are
    deliberately not evidence. Statements about what is available ("Capital One does not report gross profit", "the
    filings with text are ...") are scored not_a_claim; given the error as evidence, the judge called them supported
    on 3 of 3 claims whatever the prompt said, and with no evidence it followed the not_a_claim rule 3 of 3.
    """
    items: dict[str, str] = {}
    for t in trace:
        try:
            data = json.loads(t['result'])
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        if 'expression' in data and 'id' in data:
            items[data['id']] = f"calculation: {data['expression']} = {data.get('result')}"
        for r in data.get('results') or []:
            if not isinstance(r, dict) or 'id' not in r:
                continue
            if 'text' in r:
                items[r['id']] = f"{r.get('ticker')} {r.get('form')} {r.get('fiscal_period')} FY{r.get('fiscal_year')}, period {r.get('period')}\n{r['text']}"
            elif 'value' in r:
                extra = {k: r[k] for k in ('unit', 'derived', 'as_reported', 'segment', 'breakdown', 'source', 'period_end') if k in r}
                items[r['id']] = f"fact: {r['company']} {r['metric']} FY{r['fiscal_year']} {r['period']} = {r['value']} {json.dumps(extra)}"
    return items


GROUNDEDNESS_INSTRUCTIONS = """You check whether an answer's claims are supported by the evidence an assistant retrieved.

For each numbered claim give one verdict:
- supported: the evidence states it. Numbers must match (rounding to the precision written is fine), and the figure must be the one the claim is about: a company-wide total is NOT support for a claim about one product, segment or region.
- partial: part of the claim is supported, or it is supported only loosely.
- unsupported: no evidence supports it, or the evidence says something different.
- not_a_claim: not a statement about the filings (a greeting, a statement that a filing, period or metric is not available or not reported, a list of the filings that exist, a refusal).
Use only the evidence below, never outside knowledge. For supported and partial claims list the ids of the evidence items that support it and quote the supporting words (under 200 characters); for the others use an empty list and an empty quote.

Reply with JSON only: {"claims": [{"n": 1, "verdict": "...", "evidence_ids": ["..."], "quote": "..."}, ...]} with one entry per claim, in order."""

RELEVANCE_INSTRUCTIONS = """You judge whether an answer addresses a question. You do not know whether the answer is true.

Verdict: answers (it addresses what was asked), partial (a real part of what was asked is missing, or a substantial part of the answer is about other topics), off_topic (it does not address it).
These count as answers: a statement that the requested filing or figure is not available, with or without the most recent available filing as a fallback; a short refusal of a company or topic the system does not cover; an answer whose points are all things the sources discuss under the question's topic, even if you would not have chosen them. A sentence that is related but tangential does not make an answer partial.
You cannot check dates or whether the right filing was chosen for a relative period such as "a year ago" or "last quarter"; judge only whether the content is about what was asked.
List the numbers of any claims that do not address the question.

Reply with JSON only: {"verdict": "...", "off_topic_claims": [1, 2]}"""


def groundedness_messages(question: str, claims: list[dict], evidence: dict[str, str]) -> list[dict]:
    ev = '\n\n'.join(f'[{i}]\n{text}' for i, text in evidence.items()) or '(no evidence was retrieved)'
    cl = '\n'.join(f"{c['n']}. {c['text']}" for c in claims)
    return [{'role': 'system', 'content': GROUNDEDNESS_INSTRUCTIONS},
            {'role': 'user', 'content': f'Question: {question}\n\nEvidence:\n\n{ev}\n\nClaims:\n{cl}'}]


def relevance_messages(question: str, claims: list[dict]) -> list[dict]:
    cl = '\n'.join(f"{c['n']}. {c['text']}" for c in claims)
    return [{'role': 'system', 'content': RELEVANCE_INSTRUCTIONS},
            {'role': 'user', 'content': f'Question: {question}\n\nAnswer, as numbered claims:\n{cl}'}]


def _ask(client, model: str, messages: list[dict]) -> dict:
    response = client.chat.completions.create(model=model, temperature=0, messages=messages,
                                              response_format={'type': 'json_object'})
    return json.loads(response.choices[0].message.content)


def parse_groundedness(data: dict, claims: list[dict]) -> list[dict]:
    """Validate the judge's reply: one entry per claim, in order, with a known verdict. Raises ValueError otherwise."""
    entries = data.get('claims')
    if not isinstance(entries, list) or [e.get('n') for e in entries] != [c['n'] for c in claims]:
        raise ValueError('judge did not return one entry per claim, in order')
    out = []
    for e in entries:
        if e.get('verdict') not in VERDICTS:
            raise ValueError(f"unknown verdict {e.get('verdict')!r}")
        out.append({'n': e['n'], 'verdict': e['verdict'], 'evidence_ids': list(e.get('evidence_ids') or []),
                    'quote': e.get('quote') or ''})
    return out


def score_claim(claim: dict, verdict: dict, evidence: dict[str, str]) -> dict:
    """
    grounded: the evidence fully supports the claim. citation_correct: the claim's paragraph cites an id that is among
    the judge's supporting ids; None when the paragraph cites nothing or the claim is not a claim. Ids the judge named
    that the run never retrieved are dropped, so the judge cannot support a claim with evidence that does not exist.
    """
    supporting = [i for i in verdict['evidence_ids'] if i in evidence]
    v = verdict['verdict']
    cited = bool(claim['paragraph_ids']) and v != 'not_a_claim'
    return {**claim, 'verdict': v, 'evidence_ids': supporting, 'quote': verdict['quote'],
            'grounded': v == 'supported', 'citation_correct': (bool(set(supporting) & set(claim['paragraph_ids']))
                                                              if cited else None)}


def judge_run(record: dict, client, model: str | None = None) -> dict:
    """Judge one stored run: scored claims plus the relevance verdict. Errors in a reply raise ValueError."""
    model = model or settings.judge_model
    claims = split_claims(record['answer'])
    evidence = evidence_items(record['trace'])
    verdicts = parse_groundedness(_ask(client, model, groundedness_messages(record['question'], claims, evidence)), claims)
    rel = _ask(client, model, relevance_messages(record['question'], claims))
    if rel.get('verdict') not in RELEVANCE:
        raise ValueError(f"unknown relevance verdict {rel.get('verdict')!r}")
    return {'id': record['id'], 'run': record['run'], 'prompt_version': PROMPT_VERSION, 'model': model,
            'claims': [score_claim(c, v, evidence) for c, v in zip(claims, verdicts)],
            'relevance': rel['verdict'], 'off_topic_claims': list(rel.get('off_topic_claims') or [])}


def summarize_judged(judged: list[dict]) -> dict[str, dict]:
    """
    Per question id, from stored judge output (no model calls): grounded = supported / real claims (not_a_claim is
    left out), citation_correct = correct / claims that cite something, and how many runs got each relevance verdict.
    A rate is None when there is nothing to divide by.
    """
    by_id: dict[str, list[dict]] = {}
    for j in judged:
        by_id.setdefault(j['id'], []).append(j)
    out = {}
    for qid, runs in by_id.items():
        claims = [c for j in runs for c in j['claims'] if c['verdict'] != 'not_a_claim']
        cited = [c for c in claims if c['citation_correct'] is not None]
        out[qid] = {
            'runs': len(runs), 'claims': len(claims),
            'grounded': sum(c['grounded'] for c in claims) / len(claims) if claims else None,
            'citation_correct': sum(c['citation_correct'] for c in cited) / len(cited) if cited else None,
            'relevance': {v: sum(j['relevance'] == v for j in runs) for v in RELEVANCE},
        }
    return out


def format_judged(summary: dict[str, dict]) -> str:
    def pct(x):
        return '  n/a' if x is None else f'{x:4.0%}'
    lines = [f'{"question":38} {"claims":>6} {"grounded":>8} {"cit.ok":>6}  relevance (answers/partial/off_topic)']
    for qid, s in summary.items():
        r = s['relevance']
        lines.append(f'{qid[:38]:38} {s["claims"]:>6} {pct(s["grounded"]):>8} {pct(s["citation_correct"]):>6}  '
                     f'{r["answers"]}/{r["partial"]}/{r["off_topic"]}')
    return '\n'.join(lines)
