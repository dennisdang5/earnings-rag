"""
The blind label sheet for validating the judge: about 30 claims from stored runs, stratified so the judge is tested
on the failures we know (a company total given as a slice, unsupported figures, answer drift) and on clean answers.
The sheet shows the question, answer, claim and evidence, nothing about the judge or the stratum; those go in a
metadata file the labeller does not need. Fields to fill: `support` and `addresses_question`.
"""
import random

from earnings_rag.agent.citations import FIGURE
from earnings_rag.agent_judge import evidence_items, split_claims

CITED_CHARS = 3000      # the evidence a cited id gets in the sheet (the judge sees all of it)
OTHER_CHARS = 300       # retrieved but not cited: a preview, so the sheet stays readable
SUPPORT = ('supported', 'partial', 'unsupported', 'not_a_claim')


def candidates(records: list[dict]) -> list[dict]:
    """Every sentence of every answer, tagged with the stratum the run belongs to."""
    out = []
    for r in records:
        flagged = any(r.get(k) for k in ('uncited', 'unknown_ids', 'unsupported', 'unacknowledged_missing', 'slice_mismatch'))
        for c in split_claims(r['answer']):
            if r['id'] == 'aapl-iphone-q3-yoy' and '109,417' in c['text']:
                stratum = 'slice_error'          # the company total reported as the iPhone figure
            elif r['id'] == 'aapl-cof-competition':
                stratum = 'drift'                # lists IP and ESG items as competition risks
            elif flagged:
                stratum = 'flagged'
            elif 'do not address' in c['text'] or 'not available' in c['text'].lower():
                stratum = 'refusal'
            else:
                stratum = 'clean'
            out.append({'stratum': stratum, 'record': r, 'claim': c})
    return out


QUOTAS = {'slice_error': 2, 'flagged': 4, 'drift': 4, 'refusal': 2}   # the rest are clean


def select(records: list[dict], total: int = 30, seed: int = 7) -> list[dict]:
    """Seeded sample: the quotas above, then clean claims, never two with the same text and at most one per question
    in the clean group (spreads the sample over topics and companies); alternating dev/test halves per stratum."""
    rng = random.Random(seed)
    pool = candidates(records)
    chosen, seen_text = [], set()

    def take(stratum: str, n: int, one_per_question: bool = False):
        items = [p for p in pool if p['stratum'] == stratum and len(p['claim']['text'].split()) >= 5]
        rng.shuffle(items)
        taken, questions = 0, set()
        for p in items:
            key = p['claim']['text']
            if key in seen_text or (one_per_question and p['record']['id'] in questions):
                continue
            seen_text.add(key)
            questions.add(p['record']['id'])
            chosen.append({**p, 'split': 'dev' if taken % 2 == 0 else 'test'})
            taken += 1
            if taken == n:
                break

    for stratum, n in QUOTAS.items():
        take(stratum, n)
    take('clean', total - len(chosen), one_per_question=True)
    rng.shuffle(chosen)    # so the order of the sheet does not reveal the strata
    return chosen


def evidence_for_sheet(record: dict) -> list[dict]:
    """Evidence the answer cited in full, everything else retrieved as a short preview."""
    cited = {i for c in split_claims(record['answer']) for i in c['ids']}
    items = evidence_items(record['trace'])
    ordered = sorted(items, key=lambda i: i not in cited)
    return [{'id': i, 'cited_in_answer': i in cited,
             'text': items[i][:CITED_CHARS if i in cited else OTHER_CHARS]} for i in ordered]


def build(selection: list[dict]) -> tuple[list[dict], list[dict]]:
    """(sheet entries, metadata entries). The two share the `label_id`; only the sheet is shown to the labeller."""
    sheet, meta = [], []
    for k, s in enumerate(selection, 1):
        r, c = s['record'], s['claim']
        label_id = f'L{k:02d}'
        sheet.append({'label_id': label_id, 'question': r['question'], 'answer': r['answer'], 'claim': c['text'],
                      'cited_in_paragraph': c['paragraph_ids'], 'evidence': evidence_for_sheet(r),
                      'support': '', 'addresses_question': ''})
        meta.append({'label_id': label_id, 'stratum': s['stratum'], 'split': s['split'], 'question_id': r['id'],
                     'run': r['run'], 'claim_n': c['n'], 'has_figure': bool(FIGURE.search(c['text']))})
    return sheet, meta
