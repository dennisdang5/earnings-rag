"""
Agent eval sweep: run the same questions at several (max_steps, max_tokens) settings and compare quality with cost.
One run file per cell, so every stored run can be re-scored or judged later (eval/judge_runs.py works on a cell file).
`summarize_cell` is pure: it reads stored runs, plus the judge cache when the cell has been judged.
"""
from functools import partial
from pathlib import Path
from typing import Callable

from earnings_rag.agent.loop import run_agent
from earnings_rag.agent_eval import run_questions, summarize


def parse_grid(steps: str, tokens: str) -> list[tuple[int, int]]:
    """'3,4' and '300,500' -> every (steps, tokens) pair, steps varying slowest."""
    s = [int(x) for x in steps.split(',')]
    t = [int(x) for x in tokens.split(',')]
    if min(s) < 1 or min(t) < 1:
        raise ValueError('steps and max tokens must be positive')
    return [(a, b) for a in s for b in t]


def cell_name(max_steps: int, max_tokens: int) -> str:
    return f's{max_steps}-t{max_tokens}'


def run_sweep(questions: list[dict], grid: list[tuple[int, int]], repeats: int, client, out_dir: Path,
              run_fn: Callable = run_agent) -> dict[str, int]:
    """Run every cell into out_dir/<cell>.jsonl and return runs written per cell. A cell whose file exists is skipped,
    so an interrupted sweep resumes (delete the file to redo a cell)."""
    written = {}
    for max_steps, max_tokens in grid:
        path = out_dir / f'{cell_name(max_steps, max_tokens)}.jsonl'
        if path.exists():
            continue
        written[path.stem] = run_questions(questions, repeats, client, path,
                                           run_fn=partial(run_fn, max_steps=max_steps, max_tokens=max_tokens))
    return written


def summarize_cell(questions: list[dict], records: list[dict], judged: list[dict] | None = None) -> dict:
    """One row per cell: questions clean in every run, loop flags, mean tokens per run and, if judged, the grounded rate
    (supported / real claims), the citation-correct rate (of claims that cite) and the share of answers rated `answers`."""
    rows = summarize(questions, records)
    n_runs = sum(r['runs'] for r in rows)
    row = {
        'questions': len(rows), 'clean': sum(r['ok'] for r in rows), 'runs': n_runs,
        'cut_off': sum(r['cut_off'] for r in rows), 'truncated': sum(r['truncated'] for r in rows),
        'revised': sum(r['revised'] for r in rows),
        'input_mean': sum(r['input_mean'] * r['runs'] for r in rows) / n_runs if n_runs else 0,
        'output_mean': sum(r['output_mean'] * r['runs'] for r in rows) / n_runs if n_runs else 0,
        'grounded': None, 'citation_correct': None, 'answers': None,
    }
    if judged:
        claims = [c for j in judged for c in j['claims'] if c['verdict'] != 'not_a_claim']
        cited = [c for c in claims if c['citation_correct'] is not None]
        row['grounded'] = sum(c['grounded'] for c in claims) / len(claims) if claims else None
        row['citation_correct'] = sum(c['citation_correct'] for c in cited) / len(cited) if cited else None
        row['answers'] = sum(j['relevance'] == 'answers' for j in judged) / len(judged)
    return row


def format_sweep(cells: dict[str, dict]) -> str:
    def pct(x):
        return '   -' if x is None else f'{x:4.0%}'
    lines = [f'{"cell":10} {"clean":>7} {"cut":>4} {"trunc":>5} {"rev":>4} {"in tok":>7} {"out tok":>7} '
             f'{"grounded":>8} {"cit.ok":>6} {"answers":>7}']
    for name, c in cells.items():
        lines.append(f'{name:10} {c["clean"]:>3}/{c["questions"]:<3} {c["cut_off"]:>4} {c["truncated"]:>5} {c["revised"]:>4} '
                     f'{c["input_mean"]:>7.0f} {c["output_mean"]:>7.0f} {pct(c["grounded"]):>8} '
                     f'{pct(c["citation_correct"]):>6} {pct(c["answers"]):>7}')
    return '\n'.join(lines)
