"""
Run the agent on eval/agent_questions.yaml several times per question and store every run (live: calls the LLM).

    python eval/run_agent_eval.py [--repeats 3] [--only id1,id2] [--out eval/runs/name.jsonl]
    python eval/score_agent_eval.py [eval/runs/name.jsonl]
"""
import argparse
from datetime import datetime

from earnings_rag.agent_eval import run_questions
from earnings_rag.config import REPO_ROOT
from earnings_rag.llm import _llm
from earnings_rag.questions import load_questions

RUNS = REPO_ROOT / 'eval' / 'runs'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--repeats', type=int, default=3, help='runs per question; runs vary even at temperature 0')
    p.add_argument('--only', help='comma-separated question ids')
    p.add_argument('--out', help='run file (default: eval/runs/<timestamp>.jsonl)')
    args = p.parse_args()

    questions = load_questions(REPO_ROOT / 'eval' / 'agent_questions.yaml')
    if args.only:
        wanted = set(args.only.split(','))
        unknown = wanted - {q['id'] for q in questions}
        if unknown:
            raise SystemExit(f'unknown ids: {", ".join(sorted(unknown))}')
        questions = [q for q in questions if q['id'] in wanted]

    out = REPO_ROOT / args.out if args.out else RUNS / f'{datetime.now():%Y%m%d-%H%M%S}.jsonl'
    n = run_questions(questions, args.repeats, _llm(), out)
    print(f'{n} runs written to {out}')


if __name__ == '__main__':
    main()
