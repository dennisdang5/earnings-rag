"""
Does the relevance judge still flag answers that are NOT relevant? (live: one gpt-4o call per case)

    python eval/relevance_negatives.py [eval/runs/baseline-main.jsonl]

The human-labelled answers are all `answers`, so they cannot show whether the judge discriminates. These cases are built
from the baseline run, with a known right label:
  off_topic  a real answer shown under a different question;
  partial    a real answer with an unrelated block appended, or with half of a two-part answer cut off.
Prints the judge's verdict per case and how many match.
"""
import sys
from pathlib import Path

from earnings_rag.agent_eval import load_records
from earnings_rag.agent_judge import _ask, RELEVANCE, relevance_messages, split_claims
from earnings_rag.config import REPO_ROOT, settings
from earnings_rag.llm import _llm


def build_cases(runs: list[dict]) -> list[dict]:
    first = {}
    for r in runs:
        first.setdefault(r['id'], r)
    q = {k: v['question'] for k, v in first.items()}
    a = {k: v['answer'].strip() for k, v in first.items()}

    def keep(answer: str, marker: str) -> str:   # the part of a two-part answer before `marker`
        return answer[:answer.index(marker)].strip()

    return [
        {'name': 'revenue answer under an Apple services question', 'label': 'off_topic',
         'question': q['aapl-last-quarter'], 'answer': a['nvda-revenue-fy2022-2025']},
        {'name': 'DLSS answer under a Capital One credit risk question', 'label': 'off_topic',
         'question': q['cof-annual-report-this-year'], 'answer': a['frame-rates']},
        {'name': 'Apple services answer under an NVIDIA export controls question', 'label': 'off_topic',
         'question': q['nvda-export-controls-annual'], 'answer': a['aapl-last-quarter']},
        {'name': 'export controls answer under a Capital One gross profit question', 'label': 'off_topic',
         'question': q['cof-gross-profit-2024'], 'answer': a['nvda-export-controls-annual']},
        {'name': 'Apple services answer plus an unrelated NVIDIA export controls block', 'label': 'partial',
         'question': q['aapl-last-quarter'], 'answer': a['aapl-last-quarter'] + '\n\n' + a['nvda-export-controls-annual']},
        {'name': 'revenue answer plus an unrelated DLSS block', 'label': 'partial',
         'question': q['nvda-revenue-fy2022-2025'], 'answer': a['nvda-revenue-fy2022-2025'] + '\n\n' + a['frame-rates']},
        {'name': 'competition answer with the Capital One half removed', 'label': 'partial',
         'question': q['aapl-cof-competition'], 'answer': keep(a['aapl-cof-competition'], '### Capital One')},
        {'name': 'revenue answer with only the first two years', 'label': 'partial',
         'question': q['nvda-revenue-fy2022-2025'], 'answer': '\n'.join(a['nvda-revenue-fy2022-2025'].splitlines()[:4])},
    ]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / 'eval' / 'runs' / 'baseline-main.jsonl'
    client = _llm()
    hits = 0
    cases = build_cases(load_records(path))
    for c in cases:
        verdict = _ask(client, settings.judge_model, relevance_messages(c['question'], split_claims(c['answer'])))['verdict']
        assert verdict in RELEVANCE
        hits += verdict == c['label']
        print(f"{'ok  ' if verdict == c['label'] else 'MISS'} want {c['label']:9} got {verdict:9} {c['name']}")
    print(f'{hits}/{len(cases)} match')


if __name__ == '__main__':
    main()
