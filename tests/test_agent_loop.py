import json
from types import SimpleNamespace

from pydantic import BaseModel

from earnings_rag.agent import tools as tools_module
from datetime import date

from earnings_rag.agent import loop
from earnings_rag.agent.loop import SYSTEM_PROMPT, annotate_novelty, date_context, run_agent
from earnings_rag.agent.tools import Tool, SEARCH_FILINGS


from fakes import FakeClient, answer, tool_request


class EchoArgs(BaseModel):
    text: str

ECHO = Tool('echo', 'Echo the text back.', EchoArgs, lambda a: {'echo': a.text})

def boom(args):
    raise RuntimeError('database down')

BOOM = Tool('boom', 'Always fails.', EchoArgs, boom)

def fake_search(ids):
    """A search-shaped tool that always returns the given passage ids."""
    return Tool('search', 'Fake search.', EchoArgs, lambda a: {'results': [{'id': i} for i in ids]})


# --- loop --------------------------------------------------------------------

def test_direct_answer_uses_no_tools():
    client = FakeClient(answer('hi'))
    result = run_agent('q', client, tools=[ECHO])
    assert result.answer == 'hi'
    assert result.trace == []
    assert result.steps == 1
    assert not result.truncated

def test_tool_result_is_sent_back_to_the_model():
    client = FakeClient(tool_request('echo', '{"text": "abc"}'), answer('done'))
    result = run_agent('q', client, tools=[ECHO])

    assert result.answer == 'done'
    assert result.steps == 2
    assert result.trace[0]['tool'] == 'echo'

    second_request_messages = client.requests[1]['messages']
    assert second_request_messages[-2]['role'] == 'assistant'
    assert second_request_messages[-1] == {
        'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"echo": "abc"}',
    }

def test_invalid_arguments_become_a_tool_result_the_model_can_see():
    client = FakeClient(tool_request('echo', '{"wrong": 1}'), answer('recovered'))
    result = run_agent('q', client, tools=[ECHO])
    assert 'Invalid arguments' in json.loads(result.trace[0]['result'])['error']
    assert result.answer == 'recovered'

def test_unknown_tool_is_reported_not_raised():
    client = FakeClient(tool_request('nope', '{}'), answer('ok'))
    result = run_agent('q', client, tools=[ECHO])
    assert 'Unknown tool' in json.loads(result.trace[0]['result'])['error']

def test_tool_exception_is_returned_as_an_error():
    client = FakeClient(tool_request('boom', '{"text": "x"}'), answer('ok'))
    result = run_agent('q', client, tools=[BOOM])
    assert 'database down' in json.loads(result.trace[0]['result'])['error']

def test_step_budget_forces_an_answer_without_tools():
    client = FakeClient(
        tool_request('echo', '{"text": "a"}', 'c1'),
        tool_request('echo', '{"text": "b"}', 'c2'),
        answer('best effort'),
    )
    result = run_agent('q', client, tools=[ECHO], max_steps=2)

    assert result.truncated
    assert result.answer == 'best effort'
    assert len(result.trace) == 2
    assert 'tool_choice' not in client.requests[0]
    assert client.requests[-1]['tool_choice'] == 'none'


# --- novelty, cutoff, usage ---------------------------------------------------

def test_first_search_gets_no_note_and_counts_all_results_as_new():
    client = FakeClient(tool_request('search', '{"text": "q"}'), answer('ok'))
    result = run_agent('q', client, tools=[fake_search(['a', 'b'])])
    assert 'note' not in json.loads(result.trace[0]['result'])
    assert result.trace[0]['new_results'] == 2

def test_repeated_search_is_annotated_but_still_allowed():
    client = FakeClient(
        tool_request('search', '{"text": "q1"}', 'c1'),
        tool_request('search', '{"text": "q2"}', 'c2'),
        answer('ok'),
    )
    result = run_agent('q', client, tools=[fake_search(['a', 'b'])])
    assert result.trace[1]['new_results'] == 0
    assert '2 of 2 passages were already returned' in json.loads(result.trace[1]['result'])['note']

def test_annotate_novelty_reports_partial_overlap():
    seen = {'a'}
    out, new = annotate_novelty(json.dumps({'results': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]}), seen)
    assert new == 2
    assert '1 of 3' in json.loads(out)['note']
    assert seen == {'a', 'b', 'c'}

def test_annotate_novelty_passes_through_errors_and_other_shapes():
    for raw in ['{"error": "x"}', '{"echo": "abc"}', 'not json']:
        assert annotate_novelty(raw, set()) == (raw, None)

def test_token_usage_sums_across_all_model_calls_including_the_forced_one():
    client = FakeClient(
        tool_request('echo', '{"text": "a"}', 'c1'),
        tool_request('echo', '{"text": "b"}', 'c2'),
        answer('best effort'),
    )
    result = run_agent('q', client, tools=[ECHO], max_steps=2)
    assert len(result.usage) == 3
    assert result.input_tokens == 300
    assert result.output_tokens == 30

def test_final_answer_hitting_max_tokens_sets_cut_off():
    result = run_agent('q', FakeClient(answer('partial...', finish_reason='length')), tools=[ECHO])
    assert result.cut_off
    assert result.usage[0]['finish_reason'] == 'length'

def test_normal_answer_is_not_cut_off():
    assert not run_agent('q', FakeClient(answer('whole')), tools=[ECHO]).cut_off


# --- search_filings ----------------------------------------------------------

def test_search_filings_schema_restricts_company_to_known_tickers():
    schema = SEARCH_FILINGS.schema()['function']['parameters']
    assert schema['required'] == ['query']
    # Optional[Literal[...]] is emitted as anyOf: [the enum, null]
    variants = schema['properties']['company']['anyOf']
    assert {'enum': ['NVDA', 'AAPL', 'COF'], 'type': 'string'} in variants

def test_search_filings_passes_company_and_disables_keyword_routing(monkeypatch):
    seen = {}
    def fake_retrieve(question, **kwargs):
        seen.update(question=question, **kwargs)
        return [{'id': 'NVDA_x_0001', 'ticker': 'NVDA', 'period': 'x', 'distance': 0.123456, 'text': 'hello'}]
    monkeypatch.setattr(tools_module, 'retrieve', fake_retrieve)

    out = json.loads(SEARCH_FILINGS.call('{"query": "china exports", "company": "NVDA"}'))

    assert seen['ticker'] == 'NVDA'
    assert seen['route'] is False
    assert out['results'][0]['id'] == 'NVDA_x_0001'
    assert out['results'][0]['distance'] == 0.1235

def test_search_filings_rejects_unknown_company():
    out = json.loads(SEARCH_FILINGS.call('{"query": "x", "company": "TSLA"}'))
    assert 'error' in out


# --- calculate ---------------------------------------------------------------

def test_calculate_is_a_default_tool_with_a_required_expression():
    from earnings_rag.agent.tools import DEFAULT_TOOLS, CALCULATE
    assert CALCULATE in DEFAULT_TOOLS
    assert CALCULATE.schema()['function']['parameters']['required'] == ['expression']

def test_agent_run_through_the_real_calculate_tool():
    from earnings_rag.agent.tools import CALCULATE
    client = FakeClient(tool_request('calculate', '{"expression": "(150 - 100) / 100 * 100"}'), answer('Growth is fifty percent.'))
    result = run_agent('q', client, tools=[CALCULATE])
    assert json.loads(result.trace[0]['result'])['result'] == 50

def test_calculate_errors_reach_the_model_as_tool_results():
    from earnings_rag.agent.tools import CALCULATE
    client = FakeClient(tool_request('calculate', '{"expression": "1,000 + 5"}'), answer('ok'))
    result = run_agent('q', client, tools=[CALCULATE])
    assert 'thousands separators' in json.loads(result.trace[0]['result'])['error']


# --- date context -------------------------------------------------------------------------------------------------

def filing(ticker, form, period, fy, fp):
    return {'ticker': ticker, 'form': form, 'period': period, 'fiscal_year': fy, 'fiscal_period': fp}


def test_date_context_names_today_and_each_companys_newest_10k_and_10q():
    rows = [filing('NVDA', '10-K', '2025-01-26', 2025, 'FY'), filing('NVDA', '10-K', '2026-01-25', 2026, 'FY'),
            filing('NVDA', '10-Q', '2026-04-26', 2027, 'Q1'), filing('NVDA', '10-Q', '2026-07-26', 2027, 'Q2'),
            filing('AAPL', '10-K', '2025-09-27', 2025, 'FY')]              # oldest first, as text_periods returns
    text = date_context(date(2026, 10, 2), rows)
    assert text.startswith('Today is 2026-10-02. Newest filings with text: NVDA 10-K FY2026 (ended 2026-01-25), '
                           '10-Q Q2 FY2027 (ended 2026-07-26); AAPL 10-K FY2025 (ended 2025-09-27).')
    assert 'FY2025 (ended 2025-01-26)' not in text and 'relative dates' in text


def test_the_context_is_appended_to_the_system_prompt_and_empty_adds_nothing():
    client = FakeClient(answer('x'), answer('x'))
    run_agent('q', client, tools=[], context='Today is 2026-10-02.')
    run_agent('q', client, tools=[], context='')
    with_context, without = (r['messages'][0]['content'] for r in client.requests)
    assert with_context == SYSTEM_PROMPT + '\n\nToday is 2026-10-02.' and without == SYSTEM_PROMPT


def test_by_default_the_context_comes_from_the_database(monkeypatch):
    monkeypatch.setattr(loop, 'text_periods', lambda: [filing('COF', '10-Q', '2026-06-30', 2026, 'Q2')])
    client = FakeClient(answer('x'))
    run_agent('q', client, tools=[])
    assert 'COF 10-Q Q2 FY2026 (ended 2026-06-30)' in client.requests[0]['messages'][0]['content']
