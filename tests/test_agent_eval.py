import json
from datetime import date

import pytest

from earnings_rag.agent.loop import run_agent
from earnings_rag.agent_eval import (args_match, format_table, load_records, run_questions, score_run, summarize,
                                     REFUSAL)
from earnings_rag.config import REPO_ROOT
from earnings_rag.questions import load_questions
from tests.fakes import FakeClient, answer, tool_request


def record(**over) -> dict:
    """A run record with clean flags; tests override what they check."""
    base = {'id': 'q', 'question': 'q?', 'run': 1, 'answer': '', 'trace': [], 'cut_off': False, 'revised': False,
            'truncated': False, 'uncited': [], 'unknown_ids': [], 'unsupported': [], 'unacknowledged_missing': [],
            'input_tokens': 100, 'output_tokens': 10, 'model_calls': 1, 'tool_calls': 0}
    return {**base, **over}


def search(**args) -> dict:
    return {'step': 1, 'tool': 'search_filings', 'arguments': json.dumps({'query': 'x', **args}), 'result': '{}',
            'new_results': 1}


def test_contains_needs_every_entry_and_any_of_a_list_entry():
    expect = {'contains': ['46.21%', ['$72,880', '72.9 billion']]}
    assert score_run(expect, record(answer='46.21% and 72.9 billion'))['contains']
    assert not score_run(expect, record(answer='46.2% and 72.9 billion'))['contains']
    assert not score_run(expect, record(answer='46.21% only'))['contains']


def test_flags_and_cut_off_are_always_checked():
    assert score_run({}, record()) == {'flags': True, 'complete': True}
    assert not score_run({}, record(uncited=['a sentence']))['flags']
    assert not score_run({}, record(unsupported=[['5%', 's']]))['flags']
    assert not score_run({}, record(cut_off=True))['complete']


def test_refusal_both_directions():
    refused = record(answer=REFUSAL)
    assert score_run({'refusal': True}, refused)['refusal']
    assert not score_run({'refusal': False}, refused)['refusal']


def test_not_available_must_come_first():
    early = record(answer='The 10-Q for Q2 fiscal 2030 is not available. Filings that exist: ...')
    late = record(answer='x' * 400 + ' that filing is not available')
    assert score_run({'not_available_first': True}, early)['not_available_first']
    assert not score_run({'not_available_first': True}, late)['not_available_first']


def test_searched_matches_arguments_and_any_of_lists():
    r = record(trace=[search(company='NVDA', fiscal_year=2027, period='Q2')])
    assert args_match({'company': 'NVDA', 'period': 'Q2'}, {'period': ['Q2', 'Q3']})
    assert score_run({'searched': [{'company': 'NVDA', 'fiscal_year': 2027}]}, r)['searched[0]']
    assert not score_run({'searched': [{'company': 'AAPL'}]}, r)['searched[0]']


def test_first_search_looks_only_at_the_first_call():
    r = record(trace=[search(fiscal_year=2026), search(fiscal_year=2025)])
    assert not score_run({'first_search': {'fiscal_year': 2025}}, r)['first_search']
    assert not score_run({'first_search': {'fiscal_year': 2025}}, record())['first_search']


def test_call_count_limits():
    r = record(trace=[search(), search()])
    assert score_run({'min_calls': {'search_filings': 2}}, r)['min_calls.search_filings']
    assert not score_run({'max_calls': {'search_filings': 1}}, r)['max_calls.search_filings']
    assert score_run({'max_calls': {'calculate': 0}}, r)['max_calls.calculate']


def test_summary_reports_flaky_checks_and_cost():
    questions = [{'id': 'q', 'expect': {'contains': ['46.21%']}, 'known_issue': 'rounds'}]
    runs = [record(answer='46.21%', input_tokens=100), record(answer='46.2%', input_tokens=300, revised=True)]
    [row] = summarize(questions, runs)
    assert row['passes']['contains'] == 1 and row['runs'] == 2 and not row['ok']
    assert (row['input_mean'], row['input_max'], row['revised']) == (200, 300, 1)
    table = format_table([row])
    assert 'contains 1/2' in table and '(known issue)' in table


def test_summary_skips_questions_without_runs():
    assert summarize([{'id': 'never-run', 'expect': {}}], [record()]) == []


def test_runner_writes_one_scoreable_line_per_run(tmp_path):
    """Scripted client: one search, then an answer. today is fixed from as_of; no database (conftest stubs it)."""
    out = tmp_path / 'runs' / 'r.jsonl'
    seen = []

    def run_fn(question, client, today=None):
        seen.append(today)
        return run_agent(question, client, today=today, filings=[])

    questions = [{'id': 'q', 'question': 'what is 1 + 1?', 'as_of': '2026-10-08', 'expect': {'min_calls': {'calculate': 1}}}]
    # the same client object serves both runs, so give it enough script for two
    client = FakeClient(tool_request('calculate', '{"expression": "1 + 1"}'), answer('Two [calc_1].'),
                        tool_request('calculate', '{"expression": "1 + 1"}'), answer('Two [calc_1].'))
    assert run_questions(questions, 2, client, out, run_fn=run_fn) == 2
    assert seen == [date(2026, 10, 8)] * 2

    records = load_records(out)
    assert [r['run'] for r in records] == [1, 2]
    assert records[0]['model_calls'] == 2 and records[0]['tool_calls'] == 1
    assert records[0]['input_tokens'] == 200 and records[0]['output_tokens'] == 20
    assert score_run(questions[0]['expect'], records[0]) == {'flags': True, 'complete': True, 'min_calls.calculate': True}


def test_question_file_is_well_formed():
    questions = load_questions(REPO_ROOT / 'eval' / 'agent_questions.yaml')
    ids = [q['id'] for q in questions]
    assert len(ids) == len(set(ids))
    allowed = {'contains', 'not_contains', 'refusal', 'not_available_first', 'searched', 'first_search',
               'min_calls', 'max_calls'}
    for q in questions:
        assert q['question'] and set(q.get('expect', {})) <= allowed, q['id']
        if q.get('as_of'):
            date.fromisoformat(q['as_of'])
