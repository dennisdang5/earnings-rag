"""
Judge every run in a run file (live: gpt-4o, two calls per run) and print the claims that are not supported.

    python eval/judge_runs.py eval/runs/baseline-main.jsonl

Output is cached next to the run file as <name>.judge.jsonl, one line per (question id, run), reused while
PROMPT_VERSION is unchanged; delete the file or bump the version to re-judge.
"""
import json
import sys
from pathlib import Path

from earnings_rag.agent_eval import load_records
from earnings_rag.agent_judge import PROMPT_VERSION, judge_run
from earnings_rag.llm import _llm


def write_cache(path: Path, cache: dict) -> None:
    path.write_text(''.join(json.dumps(r) + '\n' for r in cache.values()), encoding='utf-8')


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    cache_path = path.with_suffix('.judge.jsonl')
    cache = {}
    if cache_path.exists():
        for r in load_records(cache_path):
            if r['prompt_version'] == PROMPT_VERSION:
                cache[(r['id'], r['run'])] = r
    client = _llm()
    judged = []
    for rec in load_records(path):
        key = (rec['id'], rec['run'])
        if key not in cache:
            try:
                cache[key] = judge_run(rec, client)
            except ValueError as e:
                print(f"! {key}: {e}")
                continue
            except Exception as e:   # API failure (quota, network): keep what is judged so far
                print(f"! stopped at {key}: {e}")
                break
            write_cache(cache_path, cache)
        judged.append((rec, cache[key]))
    write_cache(cache_path, cache)

    counts: dict[str, int] = {}
    for rec, j in judged:
        for c in j['claims']:
            counts[c['verdict']] = counts.get(c['verdict'], 0) + 1
            if c['verdict'] in ('unsupported', 'contradicted', 'partial'):
                print(f"\n[{rec['id']} run {rec['run']}] {c['verdict']}: {c['text']}\n  judge quote: {c['quote']!r}")
    rel: dict[str, int] = {}
    for _, j in judged:
        rel[j['relevance']] = rel.get(j['relevance'], 0) + 1
    print(f"\n{len(judged)} runs judged. claim verdicts: {counts}. relevance: {rel}")


if __name__ == '__main__':
    main()
