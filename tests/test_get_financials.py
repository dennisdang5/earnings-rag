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


def call(monkeypatch, rows=(), years=(2020, 2027), breakdowns=(), **args):
    seen = {}

    def fake_get_facts(ticker, metric, fiscal_year, fiscal_period, limit):
        seen.update(ticker=ticker, metric=metric, fiscal_year=fiscal_year, fiscal_period=fiscal_period, limit=limit)
        return list(rows)

    monkeypatch.setattr(tools_module, 'get_facts', fake_get_facts)
    monkeypatch.setattr(tools_module, 'fact_years', lambda t, m: years)
    monkeypatch.setattr(tools_module, 'available_breakdowns', lambda t, corporate: list(breakdowns))
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


# --- breakdowns --------------------------------------------------------------------------------------------------

def seg(segment, value_m, fy=2025, derived=False):
    return row(axis='product', segment=segment, value=value_m * 1e6, fiscal_year=fy, derived=derived)


NVDA_PRODUCTS = [seg('Data Center', 115_186), seg('Compute', 102_196), seg('Networking', 12_990),
                 seg('Gaming', 11_350), seg('OEM and Other', 389)]


def call_breakdown(monkeypatch, rows=(), available=(('revenue', 'product', 2021, 2026),), names=(), **args):
    seen = {}

    def fake_get_breakdown(ticker, metric, axis, fiscal_year, years):
        seen.update(ticker=ticker, metric=metric, axis=axis, fiscal_year=fiscal_year, years=years)
        return list(rows)

    monkeypatch.setattr(tools_module, 'get_breakdown', fake_get_breakdown)
    monkeypatch.setattr(tools_module, 'available_breakdowns', lambda ticker, corporate: list(available))
    monkeypatch.setattr(tools_module, 'breakdown_names', lambda ticker, metric: dict(names))
    return json.loads(GET_FINANCIALS.call(json.dumps(args))), seen


def test_breakdown_ids_name_the_axis_and_the_slice():
    r = fact_result(seg('Data Center', 115_186))
    assert r['id'] == 'NVDA_revenue_FY2025_product_DataCenter'
    assert (r['breakdown'], r['segment'], r['value']) == ('product', 'Data Center', 115_186)
    assert fact_result(seg('Corporate and other', -1))['id'].endswith('_product_CorporateAndOther')
    assert 'segment' not in fact_result(row())  # consolidated rows are unchanged


def test_breakdown_returns_every_slice_and_marks_overlapping_ones(monkeypatch):
    out, seen = call_breakdown(monkeypatch, NVDA_PRODUCTS, company='NVDA', metric='revenue', breakdown='product')
    assert seen == {'ticker': 'NVDA', 'metric': 'revenue', 'axis': 'product', 'fiscal_year': None, 'years': 2}
    part_of = {r['segment']: r.get('part_of') for r in out['results']}
    assert part_of == {'Data Center': None, 'Compute': 'Data Center', 'Networking': 'Data Center',
                       'Gaming': None, 'OEM and Other': None}
    assert 'do not add them' in out['note']


def test_breakdown_a_company_does_not_report_lists_what_it_has(monkeypatch):
    out, _ = call_breakdown(monkeypatch, available=[('revenue', 'segment', 2020, 2025)],
                            company='COF', metric='revenue', breakdown='product')
    assert 'COF does not report revenue by product' in out['error']
    assert 'revenue by segment (FY2020-FY2025)' in out['error']


def test_breakdowns_are_annual_only(monkeypatch):
    out, _ = call_breakdown(monkeypatch, company='NVDA', metric='revenue', breakdown='product', period='Q4')
    assert 'annual only' in out['error']


def test_empty_breakdown_says_which_years_exist(monkeypatch):
    out, _ = call_breakdown(monkeypatch, company='NVDA', metric='revenue', breakdown='product', fiscal_year=2012)
    assert out['results'] == [] and '2021-2026' in out['note']


def test_breakdown_result_names_the_slices_on_the_other_axes(monkeypatch):
    names = {'product': ['Data Center', 'Gaming'], 'segment': ['Compute and Networking', 'Graphics', 'Corporate and other'],
             'geography': ['US']}  # a single slice is not a breakdown, so geography is not offered
    rows = [row(axis='segment', segment='Compute and Networking', value=116_193e6)]
    out, _ = call_breakdown(monkeypatch, rows, available=[('revenue', 'segment', 2021, 2026)], names=names,
                            company='NVDA', metric='revenue', breakdown='segment')
    assert 'product: Data Center, Gaming' in out['note']
    assert 'geography' not in out['note'] and 'segment:' not in out['note']


def test_balance_rows_say_they_are_a_snapshot_on_a_date():
    r = fact_result(row(metric='deposits', fiscal_period='FY', period_start=date(2025, 12, 31),
                        period_end=date(2025, 12, 31), value=475_771e6))
    assert r['balance'] is True and r['as_of'] == '2025-12-31' and r['value'] == 475_771
    assert 'balance' not in fact_result(row())


def test_a_balance_has_no_q4_the_year_end_is_fy(monkeypatch):
    out, _ = call(monkeypatch, company='COF', metric='deposits', period='Q4')
    assert 'fiscal year end is period FY' in out['error']


def test_plain_result_lists_the_breakdowns_the_company_has(monkeypatch):
    out, _ = call(monkeypatch, rows=[row()], company='NVDA', metric='gross_profit',
                  breakdowns=[('cost_of_revenue', 'product', 2020, 2025), ('cost_of_revenue', 'segment', 2023, 2025),
                              ('revenue', 'product', 2020, 2025)])
    assert 'cost_of_revenue by product, segment; revenue by product' in out['note']
    assert 'note' not in call(monkeypatch, rows=[row()], company='NVDA', metric='revenue')[0]  # none stored: no note
