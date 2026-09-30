from earnings_rag.xbrl import normalize

M = 1_000_000


def companyfacts(*concepts):
    """concepts: (concept, unit, [rows]) tuples, in the shape of the SEC's companyfacts JSON."""
    return {'facts': {'us-gaap': {c: {'units': {unit: rows}} for c, unit, rows in concepts}}}


def fact(start, end, val, filed='2025-03-01', form='10-K', accn='0001', fy=0, fp='X'):
    return {'start': start, 'end': end, 'val': val, 'accn': accn, 'fy': fy, 'fp': fp, 'form': form, 'filed': filed}


def by_period(rows, metric):
    return {(r['fiscal_year'], r['fiscal_period']): r for r in rows if r['metric'] == metric}


# NVIDIA revenue, real figures. Fiscal 2025 ended 2025-01-26.
NVDA_REVENUE = [
    fact('2023-01-30', '2024-01-28', 60_922 * M),
    fact('2024-01-29', '2025-01-26', 130_497 * M),
    fact('2024-01-29', '2024-04-28', 26_044 * M, form='10-Q'),
    fact('2024-04-29', '2024-07-28', 30_040 * M, form='10-Q'),
    fact('2024-07-29', '2024-10-27', 35_082 * M, form='10-Q'),
    fact('2024-01-29', '2024-10-27', 91_166 * M, form='10-Q'),  # nine months year-to-date
]


def test_fiscal_year_is_the_calendar_year_it_ends_in():
    rows = normalize(companyfacts(('Revenues', 'USD', NVDA_REVENUE)), 'NVDA')
    assert by_period(rows, 'revenue')[(2025, 'FY')]['value'] == 130_497 * M


def test_q4_is_derived_as_full_year_minus_nine_months():
    rows = normalize(companyfacts(('Revenues', 'USD', NVDA_REVENUE)), 'NVDA')
    q4 = by_period(rows, 'revenue')[(2025, 'Q4')]
    assert q4['value'] == 39_331 * M
    assert q4['derived']
    assert (q4['period_start'], q4['period_end']) == ('2024-10-28', '2025-01-26')
    assert q4['accession'] == '0001'  # cites the 10-K it was derived from


def test_reported_quarters_are_not_marked_derived():
    q2 = by_period(normalize(companyfacts(('Revenues', 'USD', NVDA_REVENUE)), 'NVDA'), 'revenue')[(2025, 'Q2')]
    assert q2['value'] == 30_040 * M
    assert not q2['derived']


def test_fiscal_period_comes_from_the_facts_dates_not_the_filings_fy_field():
    # Apple's quarter ended 2024-03-30 is reported in its own 10-Q (fy=2024) and again as a prior-year
    # comparison in the next year's 10-Q (fy=2025). It is one fact and must land in fiscal 2024 Q2 only.
    apple = [
        fact('2022-09-25', '2023-09-30', 383_285 * M),
        fact('2023-10-01', '2024-09-28', 391_035 * M),
        fact('2023-12-31', '2024-03-30', 90_753 * M, form='10-Q', filed='2024-05-03', fy=2024, fp='Q2'),
        fact('2023-12-31', '2024-03-30', 90_753 * M, form='10-Q', filed='2025-05-02', fy=2025, fp='Q2'),
    ]
    rows = by_period(normalize(companyfacts(('RevenueFromContractWithCustomerExcludingAssessedTax', 'USD', apple)),
                               'AAPL'), 'revenue')
    assert (2024, 'Q2') in rows
    assert not any(fy == 2025 for fy, _ in rows)
    assert rows[(2024, 'Q2')]['filed'] == '2025-05-02'  # the later filing wins


def test_restated_value_from_the_latest_filing_wins():
    rows = [fact('2023-01-30', '2024-01-28', 100, filed='2024-02-20'),
            fact('2023-01-30', '2024-01-28', 110, filed='2025-02-20'),
            fact('2024-01-29', '2025-01-26', 120)]
    out = by_period(normalize(companyfacts(('Revenues', 'USD', rows)), 'NVDA'), 'revenue')
    assert out[(2024, 'FY')]['value'] == 110


def test_eps_is_never_derived():
    # share counts change, so FY EPS minus 9-month EPS is not Q4 EPS
    rows = [fact('2023-01-30', '2024-01-28', 1.19),
            fact('2024-01-29', '2025-01-26', 2.94),
            fact('2024-01-29', '2024-10-27', 2.04, form='10-Q')]
    out = by_period(normalize(companyfacts(('EarningsPerShareDiluted', 'USD/shares', rows)), 'NVDA'), 'eps_diluted')
    assert (2025, 'Q4') not in out
    assert out[(2025, 'FY')]['value'] == 2.94


def test_period_after_the_last_annual_fact_belongs_to_the_year_in_progress():
    rows = NVDA_REVENUE + [fact('2025-01-27', '2025-04-27', 44_062 * M, form='10-Q')]
    out = by_period(normalize(companyfacts(('Revenues', 'USD', rows)), 'NVDA'), 'revenue')
    assert out[(2026, 'Q1')]['value'] == 44_062 * M


def test_year_to_date_only_metrics_are_differenced_into_quarters():
    # cash flow is reported year-to-date only; calendar fiscal year
    rows = [fact('2023-01-01', '2023-12-31', 500),
            fact('2024-01-01', '2024-03-31', 100, form='10-Q'),
            fact('2024-01-01', '2024-06-30', 250, form='10-Q'),
            fact('2024-01-01', '2024-09-30', 420, form='10-Q'),
            fact('2024-01-01', '2024-12-31', 600)]
    out = by_period(normalize(companyfacts(('NetCashProvidedByUsedInOperatingActivities', 'USD', rows)), 'COF'),
                    'operating_cash_flow')
    assert [out[(2024, q)]['value'] for q in ('Q1', 'Q2', 'Q3', 'Q4')] == [100, 150, 170, 180]
    assert [out[(2024, q)]['derived'] for q in ('Q1', 'Q2', 'Q3', 'Q4')] == [False, True, True, True]


def test_capital_one_revenue_uses_revenues_not_the_fee_only_concept():
    cof_total = [fact('2023-01-01', '2023-12-31', 36_790 * M), fact('2024-01-01', '2024-12-31', 39_110 * M)]
    cof_fees = [fact('2023-01-01', '2023-12-31', 5_600 * M), fact('2024-01-01', '2024-12-31', 5_915 * M)]
    rows = normalize(companyfacts(('Revenues', 'USD', cof_total),
                                  ('RevenueFromContractWithCustomerExcludingAssessedTax', 'USD', cof_fees)), 'COF')
    assert by_period(rows, 'revenue')[(2024, 'FY')]['value'] == 39_110 * M


def test_metric_not_mapped_for_a_company_is_absent():
    rows = [fact('2023-01-01', '2023-12-31', 1), fact('2024-01-01', '2024-12-31', 2)]
    assert normalize(companyfacts(('GrossProfit', 'USD', rows)), 'COF') == []


def test_instant_facts_without_a_start_date_are_ignored():
    balance_sheet_style = [{'end': '2024-12-31', 'val': 1, 'accn': 'x', 'form': '10-K', 'filed': '2025-02-01'}]
    assert normalize(companyfacts(('Revenues', 'USD', balance_sheet_style)), 'NVDA') == []
