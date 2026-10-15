from datetime import date

from fakes import FakeClient, answer
from earnings_rag.agent.loop import run_agent
from earnings_rag.periods import Calendar, add_months, period_note, resolve_periods

TODAY = date(2026, 10, 2)
# newest 10-Ks of the real corpus on 2026-10-02 (text_periods() rows; the 10-Q rows do not affect the calendar)
FILINGS = [
    {'ticker': 'AAPL', 'form': '10-K', 'period': '2025-09-27', 'fiscal_year': 2025, 'fiscal_period': 'FY'},
    {'ticker': 'COF', 'form': '10-K', 'period': '2025-12-31', 'fiscal_year': 2025, 'fiscal_period': 'FY'},
    {'ticker': 'NVDA', 'form': '10-K', 'period': '2025-01-26', 'fiscal_year': 2025, 'fiscal_period': 'FY'},
    {'ticker': 'NVDA', 'form': '10-K', 'period': '2026-01-25', 'fiscal_year': 2026, 'fiscal_period': 'FY'},
    {'ticker': 'NVDA', 'form': '10-Q', 'period': '2026-07-26', 'fiscal_year': 2027, 'fiscal_period': 'Q2'},
]


def labels(question):
    return [r['label'] for r in resolve_periods(question, FILINGS, TODAY)]


def test_month_arithmetic_clamps_the_day():
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2025, 12, 31), -3) == date(2025, 9, 30)


def test_calendars_follow_each_companys_fiscal_year_end():
    nvda = Calendar(FILINGS[3])                   # FY2026 ended 2026-01-25: today is in Q3 FY2027
    assert nvda.containing(TODAY) == (2027, 3) and nvda.containing(date(2026, 1, 20)) == (2026, 4)
    apple = Calendar(FILINGS[0])                  # FY2025 ended 2025-09-27: FY2026 ended in September, today is FY2027
    assert apple.containing(TODAY) == (2027, 1) and apple.containing(date(2025, 10, 2)) == (2026, 1)


def test_next_fiscal_year_is_the_one_after_the_current_one_not_after_the_newest_10k():
    # the bug: the model read it as FY2027, the year after the newest 10-K (FY2026)
    assert labels('What did NVIDIA say in its 10-Q for the second quarter of next fiscal year?') == ['NVDA FY2028']


def test_last_quarter_is_the_last_completed_even_when_not_yet_filed():
    assert labels('What did NVIDIA say about export controls last quarter?') == ['NVDA Q2 FY2027']
    assert labels('What did Apple say last quarter?') == ['AAPL Q4 FY2026']       # ended Sept 26, no 10-K yet
    assert labels('What did Capital One say in the previous quarter?') == ['COF Q3 FY2026']


def test_this_years_annual_report_is_the_newest_10k():
    assert labels("What does Capital One say in this year's annual report?") == ['COF FY2025']
    assert labels('What does Apple say in its 10-K for this year?') == ['AAPL FY2025']


def test_this_fiscal_year_and_counted_periods():
    assert labels('Q2 of this fiscal year at Capital One?') == ['COF FY2026']
    assert labels('What did Apple say two quarters ago?') == ['AAPL Q3 FY2026']
    assert labels('What did Apple say a year ago?') == ['AAPL FY2026 (2025-10-02 falls in its Q1)']
    assert labels("Apple's annual report two years from now?") == ['AAPL FY2029 (2028-10-02 falls in its Q1)']


def test_no_company_named_resolves_for_every_company():
    assert labels('How did revenue change last quarter?') == ['AAPL Q4 FY2026', 'COF Q3 FY2026', 'NVDA Q2 FY2027']


def test_an_explicit_year_means_relative_words_are_relative_to_it():
    assert labels('How did NVIDIA revenue grow in fiscal 2025 compared with the prior year?') == []
    assert labels("NVIDIA's Q2 FY27 versus the previous quarter?") == []


def test_no_relative_period_no_note():
    assert labels('How do export controls affect NVIDIA?') == [] and period_note([], TODAY) == ''


def test_the_note_goes_beside_the_question_and_on_the_result():
    client = FakeClient(answer('x'))
    q = 'What did NVIDIA say last quarter?'
    result = run_agent(q, client, tools=[], filings=FILINGS, today=TODAY)
    assert client.requests[0]['messages'][1]['content'] == (
        q + '\n\n(Resolved from today\'s date, 2026-10-02: "last quarter" means NVDA Q2 FY2027.)')
    assert result.resolved_periods[0]['label'] == 'NVDA Q2 FY2027'
    assert 'Today is 2026-10-02' in client.requests[0]['messages'][0]['content']


def test_a_resolved_year_counts_as_asked_for_by_the_missing_filing_check():
    from pydantic import BaseModel
    from fakes import tool_request
    from earnings_rag.agent.tools import Tool

    class Args(BaseModel):
        query: str

    search = Tool('search_filings', 'Fake.', Args,
                  lambda a: {'error': 'not available', 'missing_filing': {'label': 'NVDA Q2 FY2028 10-Q', 'fiscal_year': 2028}})
    client = FakeClient(tool_request('search_filings', '{"query": "x"}'),
                        answer("NVIDIA's Q2 FY2027 10-Q discusses export controls."),
                        answer("NVIDIA's Q2 FY2027 10-Q discusses export controls."))
    result = run_agent('What did NVIDIA say in Q2 of next fiscal year?', client, tools=[search], filings=FILINGS,
                       today=TODAY)
    assert result.revised and result.unacknowledged_missing == ['NVDA Q2 FY2028 10-Q']


# --- calendar years: a bare "last/this/next year" -----------------------------------------------------------------

def test_fiscal_year_for_a_calendar_year_is_the_one_that_overlaps_it_most():
    assert Calendar(FILINGS[0]).fiscal_year_for_calendar(2025) == (2025, 270)    # Apple: Oct 2024 - Sep 2025
    assert Calendar(FILINGS[1]).fiscal_year_for_calendar(2025) == (2025, 365)    # Capital One: the same span
    assert Calendar(FILINGS[3]).fiscal_year_for_calendar(2025) == (2026, 340)    # NVIDIA: Feb 2025 - Jan 2026
    assert Calendar(FILINGS[3]).fiscal_year_for_calendar(2027) == (2028, 340)


def test_last_year_is_the_calendar_year_mapped_to_each_companys_closest_fiscal_year():
    # the bug: "last year" meant Apple FY2026, which has no 10-K yet, so answers opened with "not available"
    assert labels('What contributed the most to Apple revenue last year?') == [
        'AAPL FY2025 (2024-09-28 to 2025-09-27)']
    assert labels('How did NVIDIA do last year?') == ['NVDA FY2026 (2025-01-26 to 2026-01-25)']
    assert labels("What was Capital One's revenue last year?") == ['COF FY2025']       # same span: no caveat


def test_this_year_and_next_year_are_calendar_years_too():
    assert labels('What was Apple revenue this year?') == ['AAPL FY2026 (2025-09-28 to 2026-09-27)']
    assert labels("What is Capital One's revenue this year?") == ['COF FY2026']
    assert labels('What will NVIDIA revenue be next year?') == ['NVDA FY2028 (2027-01-26 to 2028-01-25)']


def test_fiscal_year_wording_keeps_the_fiscal_reading():
    assert labels('What did Apple report in its last fiscal year?') == ['AAPL FY2026']
    assert labels('What did NVIDIA report in the previous fiscal year?') == ['NVDA FY2026']


def test_annual_report_this_year_is_the_newest_10k_in_every_wording():
    assert labels("What does Capital One's annual report this year say about credit risk?") == ['COF FY2025']
    assert labels("What does Capital One's 10-K this year say about credit risk?") == ['COF FY2025']


def test_the_note_gives_the_fiscal_years_and_dates_and_never_says_calendar():
    resolved = resolve_periods('How did revenue change last year?', FILINGS, TODAY)
    note = period_note(resolved, TODAY)
    assert note == ("(Resolved from today's date, 2026-10-02: \"last year\" means "
                    'AAPL FY2025 (2024-09-28 to 2025-09-27), COF FY2025, NVDA FY2026 (2025-01-26 to 2026-01-25). '
                    'Say which fiscal year each figure is for.)')
    # "calendar 2025" in the note made the model fetch fiscal 2025 for NVIDIA (3 of 3 runs, DECISIONS.md), so the
    # calendar year stays on the result for the UI and out of the text the model reads
    assert 'calendar' not in note
    assert {r['calendar_year'] for r in resolved} == {2025}
    # a year that matches the company's fiscal year exactly needs no request
    assert 'Say which' not in period_note(resolve_periods("Capital One's revenue last year?", FILINGS, TODAY), TODAY)
    assert 'Say which' not in period_note(resolve_periods('What did NVIDIA say last quarter?', FILINGS, TODAY), TODAY)
