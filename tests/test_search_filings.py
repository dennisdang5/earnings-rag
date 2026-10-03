import json

from earnings_rag import pipeline
from earnings_rag.agent import tools as tools_module
from earnings_rag.agent.tools import SEARCH_FILINGS, describe_periods, missing_label
from earnings_rag.store import _where


def hit(**over):
    base = {'id': 'NVDA_2026-07-26_0043', 'ticker': 'NVDA', 'period': '2026-07-26', 'distance': 0.4139, 'text': 'hello',
            'form': '10-Q', 'fiscal_year': 2027, 'fiscal_period': 'Q2'}
    return {**base, **over}


def call(monkeypatch, hits=(), periods=(), **args):
    seen = {}

    def fake_retrieve(question, **kwargs):
        seen.update(kwargs)
        return list(hits)

    monkeypatch.setattr(tools_module, 'retrieve', fake_retrieve)
    monkeypatch.setattr(tools_module, 'text_periods', lambda ticker=None: list(periods))
    out = json.loads(SEARCH_FILINGS.call(json.dumps({'query': 'export controls', **args})))
    return out, seen


# --- which filings are searched ------------------------------------------------------------------------------------

def test_default_is_annual_reports_only_and_says_so(monkeypatch):
    out, seen = call(monkeypatch, [hit(form='10-K', fiscal_period='FY')], company='NVDA')
    assert (seen['form'], seen['fiscal_year'], seen['fiscal_period'], seen['latest']) == ('10-K', None, None, False)
    assert 'Searched annual reports (10-K) only' in out['note'] and 'latest=true' in out['note']


def test_a_quarter_searches_the_10qs_for_that_period(monkeypatch):
    out, seen = call(monkeypatch, [hit()], company='NVDA', period='Q2', fiscal_year=2027)
    assert (seen['form'], seen['fiscal_year'], seen['fiscal_period']) == ('10-Q', 2027, 'Q2')
    assert 'note' not in out                      # the model asked for a quarter: nothing to point out


def test_fy_and_a_bare_fiscal_year_search_the_annual_report(monkeypatch):
    _, seen = call(monkeypatch, [hit()], period='FY', fiscal_year=2025)
    assert (seen['form'], seen['fiscal_period']) == ('10-K', 'FY')
    out, seen = call(monkeypatch, [hit()], fiscal_year=2025)
    assert (seen['form'], seen['fiscal_year'], seen['fiscal_period']) == ('10-K', 2025, None)
    assert 'annual reports (10-K) only' in out['note']


def test_q4_is_the_annual_report_and_the_result_says_why(monkeypatch):
    out, seen = call(monkeypatch, [hit()], period='Q4', fiscal_year=2025)
    assert (seen['form'], seen['fiscal_period']) == ('10-K', 'FY')
    assert 'Q4 has no quarterly report of its own' in out['note']


def test_latest_looks_at_whichever_filing_is_newest(monkeypatch):
    out, seen = call(monkeypatch, [hit()], company='NVDA', latest=True)
    assert (seen['form'], seen['latest'], seen['fiscal_year'], seen['fiscal_period']) == (None, True, None, None)
    assert 'note' not in out


def test_latest_cannot_be_combined_with_a_period_or_year(monkeypatch):
    for extra in ({'period': 'Q1'}, {'fiscal_year': 2026}):
        out, seen = call(monkeypatch, [hit()], latest=True, **extra)
        assert 'latest cannot be combined' in out['error'] and seen == {}     # retrieve was never called


def test_unknown_period_is_rejected_by_the_schema(monkeypatch):
    out, _ = call(monkeypatch, period='Q5')
    assert 'Invalid arguments' in out['error']


# --- what comes back --------------------------------------------------------------------------------------------------

def test_results_name_the_filing_they_come_from(monkeypatch):
    out, _ = call(monkeypatch, [hit()], period='Q2')
    r = out['results'][0]
    assert (r['form'], r['fiscal_year'], r['fiscal_period'], r['period']) == ('10-Q', 2027, 'Q2', '2026-07-26')
    assert r['distance'] == 0.4139


def test_no_match_is_an_error_that_lists_the_filings_that_have_text(monkeypatch):
    periods = [{'ticker': 'NVDA', 'form': '10-K', 'period': '2023-01-29', 'fiscal_year': 2023, 'fiscal_period': 'FY'},
               {'ticker': 'NVDA', 'form': '10-K', 'period': '2026-01-25', 'fiscal_year': 2026, 'fiscal_period': 'FY'},
               {'ticker': 'NVDA', 'form': '10-Q', 'period': '2022-05-01', 'fiscal_year': 2023, 'fiscal_period': 'Q1'},
               {'ticker': 'NVDA', 'form': '10-Q', 'period': '2026-07-26', 'fiscal_year': 2027, 'fiscal_period': 'Q2'}]
    out, _ = call(monkeypatch, [], periods, company='NVDA', period='Q2', fiscal_year=2030)
    assert 'results' not in out and 'the filing asked for is not available' in out['error']
    assert 'NVDA: 10-K FY2023-FY2026 (2); 10-Q Q1 FY2023-Q2 FY2027 (2)' in out['error']
    assert out['missing_filing'] == {'label': 'NVDA Q2 FY2030 10-Q', 'fiscal_year': 2030}
    assert describe_periods([]) == ''


def test_missing_filing_labels():
    assert missing_label('NVDA', 2030, 'Q2', '10-Q') == 'NVDA Q2 FY2030 10-Q'
    assert missing_label('AAPL', 2031, 'FY', '10-K') == 'AAPL FY2031 10-K'
    assert missing_label(None, 2031, None, '10-K') == 'FY2031 10-K'
    assert missing_label('COF', None, 'Q3', '10-Q') == 'COF Q3 10-Q'
    assert missing_label('COF', None, None, None) == 'COF latest filing'


def test_schema_exposes_the_new_arguments_and_keeps_query_required():
    schema = SEARCH_FILINGS.schema()['function']['parameters']
    assert schema['required'] == ['query']
    assert {'enum': ['FY', 'Q1', 'Q2', 'Q3', 'Q4'], 'type': 'string'} in schema['properties']['period']['anyOf']
    assert schema['properties']['latest']['default'] is False
    assert 'fiscal_year' in schema['properties']


# --- the SQL for "latest" ----------------------------------------------------------------------------------------------

def test_latest_is_each_companys_newest_filing_within_the_given_form():
    where, params = _where(None, None, None, None, latest=True)
    assert where == ' WHERE period = (SELECT max(c2.period) FROM chunks c2 WHERE c2.ticker = chunks.ticker)'
    assert params == []
    where, params = _where('NVDA', '10-Q', None, None, latest=True)
    assert where == (' WHERE ticker = %s AND form = %s AND period = (SELECT max(c2.period) FROM chunks c2 '
                     'WHERE c2.ticker = chunks.ticker AND c2.form = %s)')
    assert params == ['NVDA', '10-Q', '10-Q']          # placeholders and parameters line up, in order
    assert _where('NVDA', '10-K', None, None) == (' WHERE ticker = %s AND form = %s', ['NVDA', '10-K'])   # unchanged


def test_retrieve_passes_latest_through_and_still_defaults_to_10ks(monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline, 'search', lambda vector, **kw: seen.update(kw) or [])
    pipeline.retrieve('x', query_vector=[0.0], route=False)
    assert (seen['form'], seen['latest']) == ('10-K', False)
    pipeline.retrieve('x', query_vector=[0.0], route=False, form=None, latest=True)
    assert (seen['form'], seen['latest']) == (None, True)
