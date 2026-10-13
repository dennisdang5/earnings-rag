"""
Agent eval, split into run -> store -> score. `run_questions` runs each question several times and writes one JSON
line per run (the answer, the trace, the flags, the token usage). `score_run` and `summarize` read those lines, so
scorers can change and old runs be re-scored without calling the model again. A question's `expect` block holds
checks that code can verify; whether a cited passage really supports a claim needs a judge (a later PR).
"""
import json
import re
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Callable

from earnings_rag.agent.loop import AgentResult, run_agent

REFUSAL = 'The provided filings do not address this.'
NOT_AVAILABLE = re.compile(r'not (?:yet )?(?:available|been filed|filed)|no (?:such )?(?:filing|10-[KQ])|'
                           r'does not (?:yet )?exist|has not (?:yet )?been', re.IGNORECASE)
FLAGS = ('uncited', 'unknown_ids', 'unsupported', 'unacknowledged_missing')


def to_record(question: dict, run: int, result: AgentResult) -> dict:
    """One line of a run file: the spec's id and question, plus everything the scorers and the cost table read."""
    return {'id': question['id'], 'question': question['question'], 'run': run, 'as_of': question.get('as_of'),
            **asdict(result), 'input_tokens': result.input_tokens, 'output_tokens': result.output_tokens,
            'model_calls': len(result.usage), 'tool_calls': len(result.trace)}


def run_questions(questions: list[dict], repeats: int, client, out: Path, run_fn: Callable = run_agent) -> int:
    """Run every question `repeats` times, appending each record to `out` as soon as it finishes (a crash or Ctrl-C
    keeps what ran). `as_of` fixes today's date so relative-period questions mean the same thing next month."""
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open('a', encoding='utf-8') as f:
        for q in questions:
            today = date.fromisoformat(q['as_of']) if q.get('as_of') else None
            for run in range(1, repeats + 1):
                record = to_record(q, run, run_fn(q['question'], client, today=today))
                f.write(json.dumps(record) + '\n')
                f.flush()
                count += 1
    return count


def calls(record: dict, tool: str) -> list[dict]:
    """The arguments of every call to `tool` in a run, in order (unparseable arguments count as empty)."""
    out = []
    for t in record['trace']:
        if t['tool'] == tool:
            try:
                out.append(json.loads(t['arguments']))
            except ValueError:
                out.append({})
    return out


def args_match(args: dict, want: dict) -> bool:
    """Every key in `want` equals the call's argument; a list in `want` means any of those values."""
    return all(args.get(k) in v if isinstance(v, list) else args.get(k) == v for k, v in want.items())


def score_run(expect: dict, record: dict) -> dict[str, bool]:
    """
    Pass/fail for each check in `expect`, plus two that always apply: `flags` (the loop's citation checks left
    nothing in the final answer) and `complete` (the answer was not cut off at max tokens).
    """
    answer = record['answer']
    checks = {'flags': not any(record[k] for k in FLAGS), 'complete': not record['cut_off']}

    if 'contains' in expect:   # every entry must appear; an entry that is a list is any-of (formatting variants)
        checks['contains'] = all(any(s in answer for s in ([e] if isinstance(e, str) else e)) for e in expect['contains'])
    if 'not_contains' in expect:
        checks['not_contains'] = not any(s in answer for s in expect['not_contains'])
    if 'refusal' in expect:
        checks['refusal'] = (REFUSAL in answer) == expect['refusal']
    if expect.get('not_available_first'):
        checks['not_available_first'] = bool(NOT_AVAILABLE.search(answer[:300]))
    for i, want in enumerate(expect.get('searched', [])):   # each entry needs some search_filings call matching it
        checks[f'searched[{i}]'] = any(args_match(a, want) for a in calls(record, 'search_filings'))
    if 'first_search' in expect:
        first = calls(record, 'search_filings')[:1]
        checks['first_search'] = bool(first) and args_match(first[0], expect['first_search'])
    for tool, n in expect.get('min_calls', {}).items():
        checks[f'min_calls.{tool}'] = len(calls(record, tool)) >= n
    for tool, n in expect.get('max_calls', {}).items():
        checks[f'max_calls.{tool}'] = len(calls(record, tool)) <= n
    return checks


def load_records(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(questions: list[dict], records: list[dict]) -> list[dict]:
    """
    One row per question that has runs: how many runs passed each check (a check passing in some runs and not in
    others is flaky, which a single pass/fail would hide), cost (model calls, input/output tokens: mean and max,
    since the whole history is re-sent every call), and how often the loop revised, truncated or cut off.
    """
    by_id: dict[str, list[dict]] = {}
    for r in records:
        by_id.setdefault(r['id'], []).append(r)

    rows = []
    for q in questions:
        runs = by_id.get(q['id'])
        if not runs:
            continue
        scored = [score_run(q.get('expect', {}), r) for r in runs]
        passes = {name: sum(s[name] for s in scored) for name in scored[0]}
        tokens = [r['input_tokens'] for r in runs]
        rows.append({
            'id': q['id'], 'runs': len(runs), 'passes': passes, 'known_issue': q.get('known_issue'),
            'ok': all(p == len(runs) for p in passes.values()),
            'model_calls': sum(r['model_calls'] for r in runs) / len(runs),
            'tool_calls': sum(r['tool_calls'] for r in runs) / len(runs),
            'input_mean': sum(tokens) / len(runs), 'input_max': max(tokens),
            'output_mean': sum(r['output_tokens'] for r in runs) / len(runs),
            'revised': sum(r['revised'] for r in runs), 'truncated': sum(r['truncated'] for r in runs),
            'cut_off': sum(r['cut_off'] for r in runs),
        })
    return rows


def format_table(rows: list[dict]) -> str:
    """Plain-text report: failing checks as passed/runs, then the cost columns, then totals."""
    lines = [f'{"question":38} {"runs":>4} {"calls":>5} {"in tok mean/max":>16} {"rev":>3}  result']
    for r in rows:
        failing = [f'{n} {p}/{r["runs"]}' for n, p in r['passes'].items() if p < r['runs']]
        result = 'ok' if r['ok'] else 'FAIL ' + ', '.join(failing)
        if not r['ok'] and r['known_issue']:
            result += '  (known issue)'
        lines.append(f'{r["id"][:38]:38} {r["runs"]:>4} {r["model_calls"]:>5.1f} '
                     f'{r["input_mean"]:>8.0f}/{r["input_max"]:<7} {r["revised"]:>3}  {result}')
    n = len(rows)
    passed = sum(r['ok'] for r in rows)
    known = sum(1 for r in rows if not r['ok'] and r['known_issue'])
    total = sum(r['input_mean'] * r['runs'] for r in rows)
    runs = sum(r['runs'] for r in rows)
    lines.append(f'\n{passed}/{n} questions pass every check in every run ({known} of the failures are known issues); '
                 f'mean input tokens per run {total / runs:.0f}' if runs else 'no runs')
    return '\n'.join(lines)
