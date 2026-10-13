"""
Score a stored agent run file (no model calls): pass counts per check, cost, and loop flags per question.

    python eval/score_agent_eval.py [eval/runs/name.jsonl]     # default: the newest file in eval/runs/
"""
import sys
from pathlib import Path

from earnings_rag.agent_eval import format_table, load_records, summarize
from earnings_rag.config import REPO_ROOT
from earnings_rag.questions import load_questions


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        files = sorted((REPO_ROOT / 'eval' / 'runs').glob('*.jsonl'))
        if not files:
            raise SystemExit('no run files in eval/runs/; run eval/run_agent_eval.py first')
        path = files[-1]

    questions = load_questions(REPO_ROOT / 'eval' / 'agent_questions.yaml')
    print(path)
    print(format_table(summarize(questions, load_records(path))))


if __name__ == '__main__':
    main()
