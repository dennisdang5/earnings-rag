"""
Final agreement between the judge and the labels (live: runs the judge once on the validation sheet, gpt-4o).

    python eval/final_agreement.py eval/runs/baseline-main.jsonl eval/runs/after-slice-check-v2.jsonl

Reads eval/judge_labels.yaml (the claims), eval/judge_labels_claude.yaml and eval/judge_labels_review.yaml (the human's
labels; blank entries are ignored), eval/judge_labels_meta.json (dev/test split). Writes only eval/runs/judge-final.jsonl.
"""
import json
import sys
from pathlib import Path

import yaml

from earnings_rag.agent_eval import load_records
from earnings_rag.agent_judge import PROMPT_VERSION
from earnings_rag.config import REPO_ROOT
from earnings_rag.judge_agreement import final_report, judge_sheet
from earnings_rag.llm import _llm


def load(name: str):
    return yaml.safe_load((REPO_ROOT / 'eval' / name).read_text(encoding='utf-8'))


def line(title: str, r: dict) -> None:
    print(f"{title:44} n={r['n']:<3} support exact {r['support_agree']:.0%}, supported-or-not {r['binary_agree']:.0%}, "
          f"addresses {r['addresses_agree']:.0%}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    records = [r for p in sys.argv[1:] for r in load_records(Path(p))]
    sheet = load('judge_labels.yaml')
    claude = {e['label_id']: e for e in load('judge_labels_claude.yaml')}
    human = {e['label_id']: e for e in load('judge_labels_review.yaml')}
    meta = {m['label_id']: m for m in json.loads((REPO_ROOT / 'eval' / 'judge_labels_meta.json').read_text(encoding='utf-8'))}

    judged, raw = judge_sheet(sheet, records, _llm())
    (REPO_ROOT / 'eval' / 'runs' / 'judge-final.jsonl').write_text('\n'.join(json.dumps(r) for r in raw) + '\n', encoding='utf-8')

    r = final_report(claude, human, judged, meta)
    print(f"judge prompt {PROMPT_VERSION}; human labels on {r['reviewed']} claims")
    line('Claude vs human (human-labelled claims)', r['claude_vs_human'])
    line('judge vs human (human-labelled claims)', r['judge_vs_human'])
    line('judge vs best label (all claims)', r['judge_vs_best'])
    for s, rep in r['by_split'].items():
        line(f'  {s} half', rep)
    print('judge errors against the best label (id, label source, label, judge):')
    for e in r['judge_errors']:
        print('  ', e)
    print('judge confusion vs best label (rows = label, columns = judge):')
    for a, row in r['judge_vs_best']['matrix'].items():
        print(f'  {a:12}', row)


if __name__ == '__main__':
    main()
