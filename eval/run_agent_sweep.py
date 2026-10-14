"""
Run the agent eval at every (max_steps, max_tokens) pair (live: calls the LLM), one run file per cell.

    python eval/run_agent_sweep.py --steps 4,6 --max-tokens 500,800 [--repeats 3] [--only id1,id2] [--out eval/runs/sweep-name]
    python eval/score_sweep.py eval/runs/sweep-name

Resumable: a cell whose file already exists is skipped. To add judge columns, run eval/judge_runs.py on the cell files you care about.
"""
import argparse
from datetime import datetime

from earnings_rag.agent_sweep import cell_name, parse_grid, run_sweep
from earnings_rag.config import REPO_ROOT
from earnings_rag.llm import _llm
from earnings_rag.questions import load_questions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--steps', required=True, help='comma-separated agent_max_steps values')
    p.add_argument('--max-tokens', required=True, help='comma-separated agent_max_tokens values')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--only', help='comma-separated question ids')
    p.add_argument('--out', help='output directory (default: eval/runs/sweep-<timestamp>)')
    args = p.parse_args()

    grid = parse_grid(args.steps, args.max_tokens)
    questions = load_questions(REPO_ROOT / 'eval' / 'agent_questions.yaml')
    if args.only:
        wanted = set(args.only.split(','))
        unknown = wanted - {q['id'] for q in questions}
        if unknown:
            raise SystemExit(f'unknown ids: {", ".join(sorted(unknown))}')
        questions = [q for q in questions if q['id'] in wanted]

    out = REPO_ROOT / args.out if args.out else REPO_ROOT / 'eval' / 'runs' / f'sweep-{datetime.now():%Y%m%d-%H%M%S}'
    print(f'{len(grid)} cells x {len(questions)} questions x {args.repeats} repeats = '
          f'{len(grid) * len(questions) * args.repeats} runs -> {out}')
    written = run_sweep(questions, grid, args.repeats, _llm(), out)
    skipped = [cell_name(*g) for g in grid if cell_name(*g) not in written]
    print(f'wrote {written}' + (f'; skipped existing {skipped}' if skipped else ''))


if __name__ == '__main__':
    main()
