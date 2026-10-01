from datetime import date

from earnings_rag.xbrl import normalize, find_splits, split_factor

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


# --- stock splits ------------------------------------------------------------------------------------------------

SPLIT = 'StockholdersEquityNoteStockSplitConversionRatio1'


def ratio(val, end, filed, form='10-Q'):
    return {'end': end, 'val': val, 'accn': 'x', 'fy': 0, 'fp': 'X', 'form': form, 'filed': filed}


# real rows: each split's ratio is first reported in the first filing after it, with unreliable period dates
NVDA_SPLIT_ROWS = [
    ratio(4, '2021-06-03', '2021-08-20'), ratio(4, '2021-06-03', '2021-11-22'),
    ratio(4, '2021-07-19', '2022-03-18', '10-K'), ratio(4, '2021-07-19', '2022-05-27'),
    ratio(10, '2024-05-31', '2024-08-28'), ratio(10, '2024-05-31', '2024-11-20'),
    ratio(10, '2024-06-30', '2025-05-28'),
]
AAPL_SPLIT_ROWS = [
    ratio(7, '2014-06-06', '2014-07-23'), ratio(7, '2014-06-06', '2014-10-27', '10-K'),
    ratio(4, '2020-08-28', '2020-10-30', '10-K'), ratio(4, '2020-08-28', '2021-01-28'),
]
NVDA_SPLITS = [(date(2021, 8, 20), 4), (date(2024, 8, 28), 10)]


def test_each_split_is_one_event_cut_off_at_its_earliest_filing():
    # NVIDIA's 4:1 appears with two different context dates, 46 days apart; they are still one split
    assert find_splits(companyfacts((SPLIT, 'pure', NVDA_SPLIT_ROWS))) == NVDA_SPLITS


def test_same_company_two_splits_are_distinguished_by_ratio():
    assert find_splits(companyfacts((SPLIT, 'pure', AAPL_SPLIT_ROWS))) == [(date(2014, 7, 23), 7),
                                                                           (date(2020, 10, 30), 4)]


def test_same_ratio_years_apart_is_two_splits():
    rows = [ratio(2, '2015-03-01', '2015-04-01'), ratio(2, '2020-03-01', '2020-04-01')]
    assert len(find_splits(companyfacts((SPLIT, 'pure', rows)))) == 2


def test_no_split_concept_means_no_splits():
    assert find_splits(companyfacts(('Revenues', 'USD', []))) == []


def test_split_factor_applies_only_to_values_filed_before_the_cutoff():
    assert split_factor('2024-08-27', NVDA_SPLITS) == 10       # day before: pre-split basis (only the 10:1 is after)
    assert split_factor('2024-08-28', NVDA_SPLITS) == 1        # the cutoff filing itself reports the ratio
    assert split_factor('2021-02-18', NVDA_SPLITS) == 40       # before both splits compounds
    assert split_factor('2022-01-01', NVDA_SPLITS) == 10


def nvda_eps(*rows):
    return companyfacts((SPLIT, 'pure', NVDA_SPLIT_ROWS), ('EarningsPerShareDiluted', 'USD/shares', list(rows)))


def test_eps_filed_before_a_split_is_rescaled_and_the_factor_recorded():
    rows = by_period(normalize(nvda_eps(
        fact('2021-01-31', '2022-01-30', 3.85, filed='2024-02-21'),   # as reported before the 2024 10:1 split
        fact('2022-01-31', '2023-01-29', 0.17, filed='2025-02-26'),   # already restated after it
    ), 'NVDA'), 'eps_diluted')
    assert rows[(2022, 'FY')]['value'] == 0.385
    assert rows[(2022, 'FY')]['split_factor'] == 10
    assert rows[(2023, 'FY')]['value'] == 0.17
    assert rows[(2023, 'FY')]['split_factor'] == 1


def test_adjusted_eps_makes_year_over_year_growth_sensible():
    rows = by_period(normalize(nvda_eps(
        fact('2021-01-31', '2022-01-30', 3.85, filed='2024-02-21'),
        fact('2022-01-31', '2023-01-29', 0.17, filed='2025-02-26'),
    ), 'NVDA'), 'eps_diluted')
    growth = rows[(2023, 'FY')]['value'] / rows[(2022, 'FY')]['value'] - 1
    assert -0.57 < growth < -0.55   # about -56%; unadjusted it would be -95%


def test_eps_before_both_splits_is_divided_by_both():
    rows = by_period(normalize(nvda_eps(
        fact('2020-02-03', '2021-01-31', 6.9, filed='2021-02-18'),
    ), 'NVDA'), 'eps_diluted')
    assert rows[(2021, 'FY')]['value'] == 0.1725
    assert rows[(2021, 'FY')]['split_factor'] == 40


def test_non_per_share_metrics_are_never_split_adjusted():
    rows = normalize(companyfacts((SPLIT, 'pure', NVDA_SPLIT_ROWS), ('NetIncomeLoss', 'USD', [
        fact('2021-01-31', '2022-01-30', 9_752 * M, filed='2022-03-18'),
        fact('2022-01-31', '2023-01-29', 4_368 * M, filed='2023-02-24')])), 'NVDA')
    out = by_period(rows, 'net_income')
    assert out[(2022, 'FY')]['value'] == 9_752 * M
    assert out[(2022, 'FY')]['split_factor'] == 1


def test_company_without_splits_is_unadjusted():
    rows = by_period(normalize(companyfacts(('EarningsPerShareDiluted', 'USD/shares', [
        fact('2023-01-01', '2023-12-31', 12.0, filed='2024-02-20'),
        fact('2024-01-01', '2024-12-31', 14.0, filed='2025-02-20')])), 'COF'), 'eps_diluted')
    assert [rows[(y, 'FY')]['split_factor'] for y in (2023, 2024)] == [1, 1]
