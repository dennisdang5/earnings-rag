"""
Compare the judge with labels on the validation sheet, and pick which claims a human has to adjudicate: every claim
where two labellers (here Claude and the gpt-4o judge) disagree, plus a seeded random sample of claims they agree on,
so the "both agree" cases are audited too instead of trusted.
"""
import random

from earnings_rag.agent_judge import SUPPORT_LABELS, judge_run


def find_record(entry: dict, records: list[dict]) -> dict:
    """The stored run a sheet entry came from (same question and the identical answer text)."""
    for r in records:
        if r['question'] == entry['question'] and r['answer'] == entry['answer']:
            return r
    raise KeyError(f"no stored run for {entry['label_id']}")


def judge_sheet(sheet: list[dict], records: list[dict], client, model: str | None = None) -> tuple[dict, list[dict]]:
    """
    ({label_id: {support, addresses_question, evidence_ids, grounded, citation_correct}}, raw judge_run outputs). Each
    distinct run is judged once, however many sheet claims it holds.
    """
    runs: dict[tuple, dict] = {}
    out = {}
    for entry in sheet:
        record = find_record(entry, records)
        key = (record['id'], record['run'], record['answer'])
        if key not in runs:
            runs[key] = judge_run(record, client, model)
        judged = runs[key]
        claim = next(c for c in judged['claims'] if c['text'] == entry['claim'])
        out[entry['label_id']] = {'support': claim['verdict'], 'evidence_ids': claim['evidence_ids'],
                                  'quote': claim['quote'], 'grounded': claim['grounded'],
                                  'citation_correct': claim['citation_correct'],
                                  'addresses_question': 'no' if claim['n'] in judged['off_topic_claims'] else 'yes'}
    return out, list(runs.values())


def compare(labels: dict[str, dict], judged: dict[str, dict]) -> dict:
    """Agreement between two labellers on the shared label ids: exact support label, supported-or-not, and relevance."""
    ids = sorted(set(labels) & set(judged))
    support = [(i, labels[i]['support'], judged[i]['support']) for i in ids]
    matrix = {a: {b: 0 for b in SUPPORT_LABELS} for a in SUPPORT_LABELS}
    for _, a, b in support:
        matrix[a][b] += 1
    return {
        'n': len(ids), 'matrix': matrix,
        'support_agree': sum(a == b for _, a, b in support) / len(ids),
        'binary_agree': sum((a == 'supported') == (b == 'supported') for _, a, b in support) / len(ids),
        'addresses_agree': sum(labels[i]['addresses_question'] == judged[i]['addresses_question'] for i in ids) / len(ids),
        'disagreements': [i for i in ids if labels[i]['support'] != judged[i]['support']
                          or labels[i]['addresses_question'] != judged[i]['addresses_question']],
    }


def review_set(disagreements: list[str], all_ids: list[str], agreed_sample: int = 8, seed: int = 11) -> list[str]:
    """Label ids for the human: all disagreements plus `agreed_sample` random agreed ones, in sheet order."""
    agreed = [i for i in all_ids if i not in set(disagreements)]
    chosen = set(disagreements) | set(random.Random(seed).sample(agreed, min(agreed_sample, len(agreed))))
    return [i for i in all_ids if i in chosen]
