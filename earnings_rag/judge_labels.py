"""
The blind label sheet for validating the judge: about 30 claims from stored runs, stratified so the judge is tested
on the failures we know (a company total given as a slice, unsupported figures, answer drift) and on clean answers.
The sheet shows the question, answer, claim and evidence, nothing about the judge or the stratum; those go in a
metadata file the labeller does not need. Fields to fill: `support` and `addresses_question`.
"""
import random

import re

from earnings_rag.agent.citations import FIGURE
from earnings_rag.agent_judge import evidence_items, split_claims

CITED_CHARS = 3000      # the evidence a cited id gets in the sheet (the judge sees all of it)
OTHER_CHARS = 500       # retrieved but not cited: the excerpt most similar to the claim, so the sheet stays readable
SUPPORT = ('supported', 'partial', 'unsupported', 'not_a_claim')
LABEL_HEADER = """\
# Label each claim. support: supported | partial | unsupported | not_a_claim  (is the claim backed by anything the
# agent retrieved, shown below? numbers must match, and a company total does not support a claim about one product,
# segment or region; a statement that a filing or metric is not available is backed by the tool error shown).
# addresses_question: yes | no  (does this claim help answer the question?).
# Evidence for ids cited in the answer is shown in full (cut at 3000 characters); other retrieved passages as the
# sentences most similar to the claim. The cited ids may not be the passages that support the claim: judge support
# by all the evidence shown, not by the ids cited.
"""


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


def excerpt(text: str, claim: str, chars: int = OTHER_CHARS) -> str:
    """The sentences of `text` that share the most words with the claim (in text order), up to `chars`. A fixed-length
    preview hid the support for two claims (a passage retrieved but not cited), so uncited passages show what is most
    likely to bear on the claim instead."""
    words = {w for w in re.findall(r'[a-z]{4,}', claim.lower())}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', text) if s.strip()]
    ranked = sorted(range(len(sentences)), key=lambda i: -len(words & set(re.findall(r'[a-z]{4,}', sentences[i].lower()))))
    picked, used = [], 0
    for i in ranked:
        if words & set(re.findall(r'[a-z]{4,}', sentences[i].lower())) and used + len(sentences[i]) <= chars:
            picked.append(i)
            used += len(sentences[i])
    return ' ... '.join(sentences[i] for i in sorted(picked)) or text[:200]


def evidence_for_sheet(record: dict, claim: str) -> list[dict]:
    """Evidence the answer cited in full; every other retrieved item as the excerpt most similar to the claim."""
    cited = {i for c in split_claims(record['answer']) for i in c['ids']}
    items = evidence_items(record['trace'])
    ordered = sorted(items, key=lambda i: i not in cited)
    return [{'id': i, 'cited_in_answer': i in cited,
             'text': items[i][:CITED_CHARS] if i in cited else excerpt(items[i], claim)} for i in ordered]


def build(selection: list[dict]) -> tuple[list[dict], list[dict]]:
    """(sheet entries, metadata entries). The two share the `label_id`; only the sheet is shown to the labeller."""
    sheet, meta = [], []
    for k, s in enumerate(selection, 1):
        r, c = s['record'], s['claim']
        label_id = f'L{k:02d}'
        sheet.append({'label_id': label_id, 'question': r['question'], 'answer': r['answer'], 'claim': c['text'],
                      'cited_in_paragraph': c['paragraph_ids'], 'evidence': evidence_for_sheet(r, c['text']),
                      'support': '', 'addresses_question': ''})
        meta.append({'label_id': label_id, 'stratum': s['stratum'], 'split': s['split'], 'question_id': r['id'],
                     'run': r['run'], 'claim_n': c['n'], 'has_figure': bool(FIGURE.search(c['text']))})
    return sheet, meta
