import json
from datetime import date

import pytest

from earnings_rag.agent import tools as tools_module
from earnings_rag.agent.loop import annotate_novelty
from earnings_rag.agent.tools import GET_FINANCIALS, DEFAULT_FACT_ROWS, fact_result
from fakes import FakeClient, answer, tool_request
from earnings_rag.agent.loop import run_agent


def row(**over):
    base = {'ticker': 'NVDA', 'metric': 'revenue', 'segment': '', 'fiscal_year': 2025, 'fiscal_period': 'FY',
            'concept': 'Revenues', 'unit': 'USD', 'value': 130_497_000_000.0, 'period_start': date(2024, 1, 29),
            'period_end': date(2025, 1, 26), 'derived': False, 'form': '10-K', 'accession': 'a',
            'filed': date(2025, 2, 26), 'split_factor': 1.0}
    return {**base, **over}


def call(monkeypatch, rows=(), years=(2020, 2027), **args):
    seen = {}

    def fake_get_facts(ticker, metric, fiscal_year, fiscal_period, limit):
        seen.update(ticker=ticker, metric=metric, fiscal_year=fiscal_year, fiscal_period=fiscal_period, limit=limit)
        return list(rows)

    monkeypatch.setattr(tools_module, 'get_facts', fake_get_facts)
    monkeypatch.setattr(tools_module, 'fact_years', lambda t, m: years)
    return json.loads(GET_FINANCIALS.call(json.dumps(args))), seen


def test_dollars_are_shown_in_millions_with_a_citable_id():
    r = fact_result(row())
    assert r['id'] == 'NVDA_revenue_FY2025'
    assert r['value'] == 130_497 and r['unit'] == 'USD millions'
    assert r['source'] == '10-K filed 2025-02-26'
    assert 'split_adjusted' not in r


def test_quarter_ids_carry_the_period():
    assert fact_result(row(fiscal_period='Q4', derived=True))['id'] == 'NVDA_revenue_FY2025Q4'
    assert fact_result(row(fiscal_period='Q4', derived=True))['derived'] is True


def test_split_adjusted_eps_also_gives_the_as_reported_figure():
    r = fact_result(row(metric='eps_diluted', unit='USD/shares', value=0.174, split_factor=10.0, fiscal_year=2023))
    assert r['value'] == 0.174 and r['unit'] == 'USD per share'
    assert r['split_adjusted'] is True and r['as_reported'] == 1.74


def test_unreported_metric_for_a_company_is_an_error_naming_the_alternatives(monkeypatch):
    out, _ = call(monkeypatch, company='COF', metric='gross_profit')
    assert 'COF does not report gross_profit' in out['error']
    assert 'net_interest_income' in out['error']


def test_unknown_metric_or_company_is_rejected_by_the_schema(monkeypatch):
    assert 'Invalid arguments' in call(monkeypatch, company='NVDA', metric='ebitda')[0]['error']
    assert 'Invalid arguments' in call(monkeypatch, company='TSLA', metric='revenue')[0]['error']


def test_empty_result_says_which_years_exist(monkeypatch):
    out, _ = call(monkeypatch, company='NVDA', metric='revenue', fiscal_year=2012)
    assert out['results'] == []
    assert '2020-2027' in out['note']


def test_defaults_to_the_latest_rows_and_passes_filters_through(monkeypatch):
    _, seen = call(monkeypatch, rows=[row()], company='NVDA', metric='revenue')
    assert seen == {'ticker': 'NVDA', 'metric': 'revenue', 'fiscal_year': None, 'fiscal_period': None,
                    'limit': DEFAULT_FACT_ROWS}
    _, seen = call(monkeypatch, rows=[row()], company='NVDA', metric='revenue', fiscal_year=2024, period='Q4')
    assert (seen['fiscal_year'], seen['fiscal_period']) == (2024, 'Q4')


def test_results_work_with_the_novelty_note():
    result = json.dumps({'results': [fact_result(row())]})
    seen = set()
    assert annotate_novelty(result, seen)[1] == 1
    assert annotate_novelty(result, seen)[1] == 0


def test_agent_can_chain_get_financials_into_calculate(monkeypatch):
    rows = {2024: row(fiscal_year=2024, value=60_922_000_000.0), 2025: row()}
    monkeypatch.setattr(tools_module, 'get_facts', lambda t, m, fy, fp, limit: [rows[fy]])
    client = FakeClient(
        tool_request('get_financials', '{"company": "NVDA", "metric": "revenue", "fiscal_year": 2024, "period": "FY"}', 'c1'),
        tool_request('get_financials', '{"company": "NVDA", "metric": "revenue", "fiscal_year": 2025, "period": "FY"}', 'c2'),
        tool_request('calculate', '{"expression": "(130497 - 60922) / 60922 * 100"}', 'c3'),
        answer('Revenue grew 114.2% [NVDA_revenue_FY2024] [NVDA_revenue_FY2025].'),
    )
    result = run_agent('NVIDIA revenue growth?', client)
    assert [t['tool'] for t in result.trace] == ['get_financials', 'get_financials', 'calculate']
    assert '114.2' in json.loads(result.trace[2]['result'])['result'].__str__()
    assert 'NVDA_revenue_FY2025' in result.answer
