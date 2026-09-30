import json
from types import SimpleNamespace

from pydantic import BaseModel

from earnings_rag.agent import tools as tools_module
from earnings_rag.agent.loop import run_agent
from earnings_rag.agent.tools import Tool, SEARCH_FILINGS


# --- scripted fake of the OpenAI client -------------------------------------

def answer(text):
    return SimpleNamespace(content=text, tool_calls=None)

def tool_request(name, arguments, call_id='call_1'):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(content=None, tool_calls=[call])

class FakeClient:
    """Returns the scripted messages in order and records every request it receives."""
    def __init__(self, *messages):
        self.script = list(messages)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.script.pop(0))])


class EchoArgs(BaseModel):
    text: str

ECHO = Tool('echo', 'Echo the text back.', EchoArgs, lambda a: {'echo': a.text})

def boom(args):
    raise RuntimeError('database down')

BOOM = Tool('boom', 'Always fails.', EchoArgs, boom)


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
