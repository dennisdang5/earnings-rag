import json
import threading
from functools import partial

from pydantic import BaseModel

from earnings_rag.agent.loop import run_agent
from earnings_rag.agent.tools import Tool
from earnings_rag.agent_stream import stream_events, summarize_result, to_stream_event
from fakes import FakeClient, answer, tool_request


class Args(BaseModel):
    text: str


def facts(a):
    return {'results': [{'id': 'NVDA_revenue_FY2025', 'value': 130497.0}]}


FACTS = Tool('get', 'Return a fact.', Args, facts)
RUN = partial(run_agent, tools=[FACTS])


def collect(client, run_fn=RUN):
    return list(stream_events('q', client, run_fn=run_fn))


def test_a_run_streams_its_steps_then_the_checked_answer_whole():
    client = FakeClient(tool_request('get', '{"text": "x"}'),
                        answer('Revenue was $130.5 billion.'),
                        answer('Revenue was $130.5 billion [NVDA_revenue_FY2025].'))
    out = collect(client)
    names = [n for n, _ in out]
    assert names == ['status', 'tool_call', 'tool_result', 'status', 'revising', 'status', 'answer', 'details', 'done']
    assert dict(out)['tool_result']['ids'] == ['NVDA_revenue_FY2025']
    # only the revised answer is ever sent; the draft that failed the checks is not
    assert [d['text'] for n, d in out if n == 'answer'] == ['Revenue was $130.5 billion [NVDA_revenue_FY2025].']
    details = dict(out)['details']
    assert details['revised'] is True and len(details['trace']) == 1 and len(details['usage']) == 3
    assert details['trace'][0]['result'] and details['uncited'] == []


def test_the_forced_final_call_is_a_writing_status():
    client = FakeClient(tool_request('get', '{"text": "a"}'), answer('best effort'))
    out = collect(client, partial(run_agent, tools=[FACTS], max_steps=1))
    assert ('status', {'stage': 'writing', 'step': 2}) in out and dict(out)['details']['truncated'] is True


def test_a_failure_yields_an_error_event_and_is_then_raised():
    def broken(question, client, on_event=None):
        on_event({'type': 'start', 'resolved_periods': []})
        raise RuntimeError('database down')

    seen = []
    gen = stream_events('q', None, run_fn=broken)
    try:
        for item in gen:
            seen.append(item[0])
    except RuntimeError as e:
        assert str(e) == 'database down'
    else:
        raise AssertionError('the failure should be re-raised after the error event')
    assert seen == ['error']


def test_closing_the_stream_stops_the_run_at_its_next_event():
    gate = threading.Event()
    reached = []

    def slow(question, client, on_event=None):
        on_event({'type': 'model_call', 'step': 1})
        gate.wait(5)
        on_event({'type': 'model_call', 'step': 2})   # raises Cancelled once the stream is closed
        reached.append('past the cancelled event')   # must not run
        return None

    gen = stream_events('q', None, run_fn=slow)
    assert next(gen)[0] == 'status'
    gen.close()
    gate.set()
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(2)
    assert reached == []


def test_result_summaries_and_event_mapping():
    assert summarize_result(json.dumps({'results': [{'id': 'a'}, {'id': 'b'}]})) == {'ids': ['a', 'b'], 'count': 2, 'error': None}
    assert summarize_result(json.dumps({'id': 'calc_1', 'expression': '1+1', 'result': 2}))['ids'] == ['calc_1']
    assert summarize_result(json.dumps({'error': 'not available'})) == {'ids': [], 'count': 0, 'error': 'not available'}
    assert summarize_result('not json') == {'ids': [], 'count': 0, 'error': None}
    assert to_stream_event({'type': 'start', 'resolved_periods': []}) is None


def test_endpoint_sends_sse_events(monkeypatch):
    from fastapi.testclient import TestClient
    from earnings_rag import api

    client = FakeClient(tool_request('get', '{"text": "x"}'), answer('Revenue was $130.5 billion [NVDA_revenue_FY2025].'))
    monkeypatch.setattr(api, 'get_agent_client', lambda: client)
    monkeypatch.setattr(api, 'stream_events', lambda q, c: stream_events(q, c, run_fn=RUN))
    response = TestClient(api.app).post('/agent/stream', json={'question': 'NVIDIA revenue?'})
    assert response.status_code == 200 and response.headers['content-type'].startswith('text/event-stream')
    events = [line[len('event: '):] for line in response.text.splitlines() if line.startswith('event: ')]
    assert events == ['status', 'tool_call', 'tool_result', 'status', 'answer', 'details', 'done']
    assert TestClient(api.app).post('/agent/stream', json={'question': ''}).status_code == 422


def test_details_carry_the_ingest_decisions_for_the_metrics_the_run_read():
    class FinArgs(BaseModel):
        company: str
        metric: str

    fin = Tool('get_financials', 'Return facts.', FinArgs, lambda a: {'results': [{'id': 'NVDA_revenue_FY2025', 'value': 1.0}]})
    seen = []

    def lookup(pairs):
        seen.append(pairs)
        return [{'ticker': 'NVDA', 'metric': 'revenue', 'kind': 'stale', 'detail': 'x'}]

    client = FakeClient(tool_request('get_financials', '{"company": "NVDA", "metric": "revenue"}'),
                        answer('Revenue was $1 million [NVDA_revenue_FY2025].'))
    out = dict(stream_events('q', client, run_fn=partial(run_agent, tools=[fin]), decisions_fn=lookup))
    assert seen == [{('NVDA', 'revenue')}]
    assert out['details']['ingest_decisions'][0]['kind'] == 'stale'


def test_a_failing_decisions_lookup_does_not_lose_the_answer():
    def broken(pairs):
        raise RuntimeError('db down')

    from earnings_rag.agent_stream import ingest_decisions_for
    trace = [{'tool': 'get_financials', 'arguments': {'company': 'AAPL', 'metric': 'revenue'}}]
    assert ingest_decisions_for(trace, broken) == []
    assert ingest_decisions_for([{'tool': 'search_filings', 'arguments': {}}], broken) == []
