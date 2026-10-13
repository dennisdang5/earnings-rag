"""
Run the judge on the validation sheet's claims and compare it with another labeller's labels (live: calls gpt-4o).

    python eval/validate_judge.py eval/judge_labels_claude.yaml eval/runs/baseline-main.jsonl eval/runs/after-slice-check-v2.jsonl

Writes eval/runs/judge-validation.jsonl (raw judge output), prints agreement and the disagreements, and writes
eval/judge_labels_review.yaml: the claims a human still has to label (disagreements plus a random sample of agreements),
with blank label fields and nothing about who said what.
"""
import json
import sys
from pathlib import Path

import yaml

from earnings_rag.agent_eval import load_records
from earnings_rag.config import REPO_ROOT
from earnings_rag.judge_agreement import compare, judge_sheet, review_set
from earnings_rag.judge_labels import LABEL_HEADER
from earnings_rag.llm import _llm


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    labels = {e['label_id']: e for e in yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8'))}
    records = [r for p in sys.argv[2:] for r in load_records(Path(p))]
    sheet = yaml.safe_load((REPO_ROOT / 'eval' / 'judge_labels.yaml').read_text(encoding='utf-8'))

    judged, raw = judge_sheet(sheet, records, _llm())
    out = REPO_ROOT / 'eval' / 'runs' / 'judge-validation.jsonl'
    out.write_text('\n'.join(json.dumps(r) for r in raw) + '\n', encoding='utf-8')

    report = compare(labels, judged)
    print(f"{report['n']} claims: exact support agreement {report['support_agree']:.0%}, supported-or-not "
          f"{report['binary_agree']:.0%}, addresses-question {report['addresses_agree']:.0%}")
    print('rows = labeller, columns = judge')
    for a, row in report['matrix'].items():
        print(f'  {a:12}', row)
    by_id = {e['label_id']: e for e in sheet}
    for i in report['disagreements']:
        print(f"\n{i}: labeller {labels[i]['support']}/{labels[i]['addresses_question']}  judge "
              f"{judged[i]['support']}/{judged[i]['addresses_question']}\n  claim: {by_id[i]['claim'][:150]}\n  judge quote: "
              f"{judged[i]['quote'][:150]}\n  labeller note: {labels[i].get('note', '')}")

    chosen = review_set(report['disagreements'], [e['label_id'] for e in sheet])
    review = [{**e, 'support': '', 'addresses_question': ''} for e in sheet if e['label_id'] in chosen]
    (REPO_ROOT / 'eval' / 'judge_labels_review.yaml').write_text(
        LABEL_HEADER + yaml.safe_dump(review, sort_keys=False, allow_unicode=True, width=120), encoding='utf-8')
    print(f'\n{len(review)} claims for the human to label -> eval/judge_labels_review.yaml '
          f'({len(report["disagreements"])} disagreements + {len(review) - len(report["disagreements"])} agreed)')


if __name__ == '__main__':
    main()
