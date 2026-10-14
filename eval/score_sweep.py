"""
Compare the cells of a sweep directory (no model calls): clean questions, loop flags, tokens, and the judge columns
for cells that have a <cell>.judge.jsonl (from eval/judge_runs.py).

    python eval/score_sweep.py eval/runs/sweep-name
"""
import sys
from pathlib import Path

from earnings_rag.agent_eval import load_records
from earnings_rag.agent_judge import PROMPT_VERSION
from earnings_rag.agent_sweep import format_sweep, summarize_cell
from earnings_rag.config import REPO_ROOT
from earnings_rag.questions import load_questions


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    directory = Path(sys.argv[1])
    questions = load_questions(REPO_ROOT / 'eval' / 'agent_questions.yaml')
    cells = {}
    for path in sorted(directory.glob('s*-t*.jsonl')):
        if path.name.endswith('.judge.jsonl'):
            continue
        judge_path = path.with_suffix('.judge.jsonl')
        judged = [j for j in load_records(judge_path) if j['prompt_version'] == PROMPT_VERSION] if judge_path.exists() else None
        cells[path.stem] = summarize_cell(questions, load_records(path), judged)
    print(directory)
    print(format_sweep(cells))


if __name__ == '__main__':
    main()
