from pydantic import BaseModel

from fakes import FakeClient, answer, tool_request
from earnings_rag.agent.citations import check_missing_filings, revision_request
from earnings_rag.agent.loop import missing_filing, run_agent
from earnings_rag.agent.tools import Tool

Q = 'What did NVIDIA say about export controls in its 10-Q for the second quarter of fiscal 2030?'
MISSING = [{'label': 'NVDA Q2 FY2030 10-Q', 'fiscal_year': 2030}]

# the two answers seen in live runs: a hit (stopped after the error) and a miss (searched on and answered from Q2 FY2027)
ADMITS = ("NVIDIA's 10-Q for the second quarter of fiscal 2030 is not available. However, I can provide information "
          "from other filings.")
SILENT = ("NVIDIA's most recent 10-Q, which is for Q2 of fiscal 2027, discusses export controls in detail "
          "[NVDA_2026-07-26_0043].")


# --- the check ----------------------------------------------------------------------------------------------------

def test_an_answer_that_admits_the_filing_is_missing_passes():
    assert check_missing_filings(ADMITS, Q, MISSING) == []


def test_an_answer_from_another_filing_that_does_not_say_so_is_flagged():
    assert check_missing_filings(SILENT, Q, MISSING) == ['NVDA Q2 FY2030 10-Q']


def test_other_ways_of_saying_it_pass():
    for text in ('There is no 10-Q for Q2 FY2030 in the filings.',
                 "NVIDIA's Q2 fiscal 2030 report doesn't exist yet.",
                 'I could not find a 10-Q for fiscal 2030.',
                 'You asked about Q2 FY2030. That filing has not been filed yet.'):    # year in the sentence before
        assert check_missing_filings(text, Q, MISSING) == [], text


def test_the_year_and_the_phrase_must_be_together():
    text = 'Q2 FY2030 is a future quarter. Export controls are discussed below. Some data is not available.'
    assert check_missing_filings(text, Q, MISSING) == ['NVDA Q2 FY2030 10-Q']


def test_a_period_the_user_did_not_ask_for_is_not_checked():
    # the model guessed fiscal 2028 for "the latest quarter", got the error, and recovered with latest=true
    assert check_missing_filings(SILENT, 'What did NVIDIA say in its latest 10-Q?',
                                 [{'label': 'NVDA Q3 FY2028 10-Q', 'fiscal_year': 2028}]) == []


def test_a_missing_filing_without_a_year_is_not_checked_and_repeats_count_once():
    assert check_missing_filings(SILENT, Q, [{'label': 'COF latest filing', 'fiscal_year': None}]) == []
    assert check_missing_filings(SILENT, Q, MISSING * 2) == ['NVDA Q2 FY2030 10-Q']


def test_the_revision_request_names_the_filing_and_what_to_do():
    text = revision_request({'uncited': [], 'unknown_ids': [], 'unacknowledged_missing': ['NVDA Q2 FY2030 10-Q']})
    assert 'NVDA Q2 FY2030 10-Q' in text and 'Begin the answer by saying the filing asked for is not available' in text
    assert 'figure' not in text


def test_missing_filing_is_read_from_the_result_only():
    assert missing_filing('{"error": "x", "missing_filing": {"label": "L", "fiscal_year": 2030}}')['label'] == 'L'
    for other in ('{"results": []}', '[1, 2]', 'not json', '"a string"'):
        assert missing_filing(other) is None


# --- in the loop --------------------------------------------------------------------------------------------------

class Args(BaseModel):
    query: str


def fake_search(args):
    if args.query == 'q2 fy2030':
        return {'error': 'not available', 'missing_filing': MISSING[0]}
    return {'results': [{'id': 'NVDA_2026-07-26_0043'}]}


SEARCH = Tool('search_filings', 'Fake search.', Args, fake_search)


def run(*script, question=Q):
    client = FakeClient(*script)
    return run_agent(question, client, tools=[SEARCH]), client


def test_error_then_another_period_then_silence_is_sent_back_once():
    result, client = run(tool_request('search_filings', '{"query": "q2 fy2030"}'),
                         tool_request('search_filings', '{"query": "latest"}', 'call_2'),
                         answer(SILENT),
                         answer(ADMITS + ' ' + SILENT))
    assert result.revised and result.unacknowledged_missing == []
    sent_back = client.requests[3]['messages'][-1]
    assert sent_back['role'] == 'user' and 'NVDA Q2 FY2030 10-Q' in sent_back['content']


def test_an_answer_that_admits_it_costs_no_extra_call():
    result, client = run(tool_request('search_filings', '{"query": "q2 fy2030"}'), answer(ADMITS))
    assert not result.revised and len(client.requests) == 2


def test_a_second_silent_answer_is_accepted_and_recorded():
    result, client = run(tool_request('search_filings', '{"query": "q2 fy2030"}'), answer(SILENT), answer(SILENT))
    assert result.revised and result.unacknowledged_missing == ['NVDA Q2 FY2030 10-Q'] and len(client.requests) == 3


def test_a_citation_problem_and_a_missing_filing_share_one_revision():
    result, client = run(tool_request('search_filings', '{"query": "q2 fy2030"}'),
                         answer('Revenue was $5 billion [NVDA_9999_0001].'),
                         answer('Revenue was $5 billion [NVDA_9999_0001].'))
    request = client.requests[2]['messages'][-1]['content']
    assert 'never returned by a tool' in request and 'NVDA Q2 FY2030 10-Q' in request
    assert result.unknown_ids == ['NVDA_9999_0001'] and result.unacknowledged_missing == ['NVDA Q2 FY2030 10-Q']
    assert len(client.requests) == 3                  # one revision, not one per problem


def test_the_forced_answer_is_checked_but_not_revised():
    client = FakeClient(tool_request('search_filings', '{"query": "q2 fy2030"}'), answer(SILENT))
    result = run_agent(Q, client, tools=[SEARCH], max_steps=1)
    assert result.truncated and not result.revised and result.unacknowledged_missing == ['NVDA Q2 FY2030 10-Q']
