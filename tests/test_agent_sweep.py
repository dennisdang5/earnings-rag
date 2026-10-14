import pytest

from earnings_rag.agent.loop import run_agent
from earnings_rag.agent_eval import load_records
from earnings_rag.agent_sweep import cell_name, format_sweep, parse_grid, run_sweep, summarize_cell
from fakes import FakeClient, answer, tool_request


def test_parse_grid_is_every_pair_steps_slowest():
    assert parse_grid('3,6', '300,500') == [(3, 300), (3, 500), (6, 300), (6, 500)]
    assert cell_name(6, 500) == 's6-t500'
    with pytest.raises(ValueError):
        parse_grid('0', '500')


def test_sweep_passes_each_cell_its_settings_and_skips_existing_files(tmp_path):
    seen = []

    def run_fn(question, client, today=None, max_steps=None, max_tokens=None):
        seen.append((max_steps, max_tokens))
        return run_agent(question, client, today=today, filings=[], slices={}, max_steps=max_steps, max_tokens=max_tokens)

    questions = [{'id': 'q', 'question': 'hi', 'expect': {}}]
    grid = [(2, 100), (4, 200)]
    written = run_sweep(questions, grid, 1, FakeClient(answer('a'), answer('b')), tmp_path, run_fn=run_fn)
    assert written == {'s2-t100': 1, 's4-t200': 1} and seen == [(2, 100), (4, 200)]
    assert len(load_records(tmp_path / 's4-t200.jsonl')) == 1

    # a second sweep over a larger grid only runs the new cell
    seen.clear()
    written = run_sweep(questions, grid + [(6, 300)], 1, FakeClient(answer('c')), tmp_path, run_fn=run_fn)
    assert written == {'s6-t300': 1} and seen == [(6, 300)]


def test_steps_setting_changes_how_the_run_ends(tmp_path):
    """With one step the tool call uses the whole budget, so the answer is forced and marked truncated."""
    def run_fn(question, client, today=None, max_steps=None, max_tokens=None):
        return run_agent(question, client, today=today, filings=[], slices={}, max_steps=max_steps, max_tokens=max_tokens)

    questions = [{'id': 'q', 'question': 'what is 1 + 1?', 'expect': {}}]
    client = FakeClient(tool_request('calculate', '{"expression": "1 + 1"}'), answer('Two [calc_1].'))
    run_sweep(questions, [(1, 500)], 1, client, tmp_path, run_fn=run_fn)
    assert load_records(tmp_path / 's1-t500.jsonl')[0]['truncated'] is True


def claim(verdict, cc=None):
    return {'verdict': verdict, 'grounded': verdict == 'supported', 'citation_correct': cc}


def test_summarize_cell_with_and_without_judge():
    questions = [{'id': 'q', 'expect': {}}]
    base = {'id': 'q', 'question': 'q?', 'answer': '', 'trace': [], 'cut_off': False, 'revised': False, 'truncated': False,
            'uncited': [], 'unknown_ids': [], 'unsupported': [], 'unacknowledged_missing': [], 'slice_mismatch': [],
            'model_calls': 1, 'tool_calls': 0}
    records = [{**base, 'run': 1, 'input_tokens': 100, 'output_tokens': 10},
               {**base, 'run': 2, 'input_tokens': 300, 'output_tokens': 30, 'cut_off': True}]
    row = summarize_cell(questions, records)
    assert (row['clean'], row['questions'], row['cut_off'], row['input_mean'], row['output_mean']) == (0, 1, 1, 200, 20)
    assert row['grounded'] is None and 'cell' in format_sweep({'s6-t500': row}) and '   -' in format_sweep({'s6-t500': row})

    judged = [{'claims': [claim('supported', True), claim('unsupported', False), claim('not_a_claim')], 'relevance': 'answers'},
              {'claims': [claim('supported', None)], 'relevance': 'partial'}]
    row = summarize_cell(questions, records, judged)
    assert row['grounded'] == 2 / 3 and row['citation_correct'] == 0.5 and row['answers'] == 0.5
