"""
Agreement between the judge's answer-level relevance verdicts and the labels (no model calls: reads the judge cache).

    python eval/relevance_agreement.py [eval/runs/baseline-main.jsonl]

Needs <run file>.judge.jsonl (eval/judge_runs.py), eval/relevance_labels_claude.yaml and eval/relevance_labels_review.yaml
(the human's labels; blank entries are ignored). Prints judge vs human, judge vs Claude, Claude vs human, and each disagreement.
"""
import sys
from pathlib import Path

import yaml

from earnings_rag.agent_eval import load_records
from earnings_rag.agent_judge import PROMPT_VERSION
from earnings_rag.config import REPO_ROOT


def agreement(a: dict, b: dict) -> tuple[int, int, list]:
    """Compare two {key: label} maps on the keys both have (blank labels count as missing)."""
    keys = [k for k in a if a[k] and b.get(k)]
    diff = [k for k in keys if a[k] != b[k]]
    return len(keys) - len(diff), len(keys), diff


def load_labels(name: str) -> dict:
    path = REPO_ROOT / 'eval' / name
    return {(e['id'], e['run']): e.get('label') for e in yaml.safe_load(path.read_text(encoding='utf-8'))}


def main() -> None:
    run_file = Path(sys.argv[1] if len(sys.argv) > 1 else REPO_ROOT / 'eval' / 'runs' / 'baseline-main.jsonl')
    judged = [j for j in load_records(run_file.with_suffix('.judge.jsonl')) if j['prompt_version'] == PROMPT_VERSION]
    judge = {(j['id'], j['run']): j['relevance'] for j in judged}
    claude, human = load_labels('relevance_labels_claude.yaml'), load_labels('relevance_labels_review.yaml')
    for title, a, b in [('judge vs human', judge, human), ('judge vs Claude', judge, claude), ('Claude vs human', claude, human)]:
        ok, n, diff = agreement(a, b)
        print(f'{title:16} {ok}/{n}' + (f' ({ok / n:.0%})' if n else ''))
        for k in diff:
            print(f'    {k[0]} run {k[1]}: {title.split(" vs ")[0]}={a[k]}, {title.split(" vs ")[1]}={b[k]}')


if __name__ == '__main__':
    main()
