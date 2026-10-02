from datetime import date

import pytest

from earnings_rag import xbrl
from earnings_rag.xbrl import normalize, find_splits, split_factor, check_split_consistency

M = 1_000_000


def companyfacts(*concepts, cover=None):
    """
    concepts: (concept, unit, [rows]) tuples, in the shape of the SEC's companyfacts JSON.
    cover: [(filing date, shares outstanding on the cover page)], one per filing.
    """
    facts = {'us-gaap': {c: {'units': {unit: rows}} for c, unit, rows in concepts}}
    if cover:
        rows = [{'end': filed, 'val': v, 'accn': f'a{i}', 'form': '10-Q', 'filed': filed}
                for i, (filed, v) in enumerate(cover)]
        facts['dei'] = {'EntityCommonStockSharesOutstanding': {'units': {'shares': rows}}}
    return {'facts': facts}


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
    # two revenue synonyms disagree; the larger is confirmed because net interest + non-interest income equals it
    cof_total = [fact('2023-01-01', '2023-12-31', 36_790 * M), fact('2024-01-01', '2024-12-31', 39_110 * M)]
    cof_fees = [fact('2023-01-01', '2023-12-31', 5_600 * M), fact('2024-01-01', '2024-12-31', 5_915 * M)]
    nii = [fact('2023-01-01', '2023-12-31', 29_000 * M), fact('2024-01-01', '2024-12-31', 31_200 * M)]
    nonii = [fact('2023-01-01', '2023-12-31', 7_790 * M), fact('2024-01-01', '2024-12-31', 7_910 * M)]
    rows = normalize(companyfacts(('Revenues', 'USD', cof_total),
                                  ('RevenueFromContractWithCustomerExcludingAssessedTax', 'USD', cof_fees),
                                  ('InterestIncomeExpenseNet', 'USD', nii), ('NoninterestIncome', 'USD', nonii)), 'COF')
    revenue = by_period(rows, 'revenue')[(2024, 'FY')]
    assert revenue['value'] == 39_110 * M and revenue['concept'] == 'Revenues'


def test_disagreeing_revenue_synonyms_that_no_identity_confirms_are_not_stored():
    rows = normalize(companyfacts(
        ('Revenues', 'USD', [fact('2023-01-01', '2023-12-31', 100 * M), fact('2024-01-01', '2024-12-31', 120 * M)]),
        ('RevenueFromContractWithCustomerExcludingAssessedTax', 'USD',
         [fact('2023-01-01', '2023-12-31', 90 * M), fact('2024-01-01', '2024-12-31', 110 * M)])), 'XYZ')
    assert 'revenue' not in {r['metric'] for r in rows}


def test_a_company_that_switched_revenue_concepts_keeps_every_year_and_names_the_concept():
    old = [fact('2022-01-01', '2022-12-31', 80 * M)]
    new = [fact('2023-01-01', '2023-12-31', 100 * M), fact('2024-01-01', '2024-12-31', 120 * M)]
    out = by_period(normalize(companyfacts(('SalesRevenueNet', 'USD', old),
                                           ('RevenueFromContractWithCustomerExcludingAssessedTax', 'USD', new)),
                              'XYZ'), 'revenue')
    assert [out[(fy, 'FY')]['value'] // M for fy in (2022, 2023, 2024)] == [80, 100, 120]
    assert out[(2022, 'FY')]['concept'] == 'SalesRevenueNet'
    assert out[(2024, 'FY')]['concept'] == 'RevenueFromContractWithCustomerExcludingAssessedTax'


def test_metric_with_no_concept_in_the_data_is_absent():
    rows = [fact('2023-01-01', '2023-12-31', 1), fact('2024-01-01', '2024-12-31', 2)]
    assert 'gross_profit' not in {r['metric'] for r in normalize(companyfacts(('Revenues', 'USD', rows)), 'COF')}


def test_instant_facts_without_a_start_date_are_ignored():
    balance_sheet_style = [{'end': '2024-12-31', 'val': 1, 'accn': 'x', 'form': '10-K', 'filed': '2025-02-01'}]
    assert normalize(companyfacts(('Revenues', 'USD', balance_sheet_style)), 'NVDA') == []


# --- stock splits ------------------------------------------------------------------------------------------------
# Detected from the share count on each filing's cover page, which jumps by the split ratio.

# NVIDIA's real pattern: 4:1 effective July 2021 and 10:1 effective June 2024
NVDA_COVER = [
    ('2021-02-18', 620 * M), ('2021-05-26', 618 * M),
    ('2021-08-20', 2_490 * M), ('2021-11-22', 2_500 * M),      # first filing after the 4:1
    ('2024-02-21', 2_470 * M), ('2024-05-29', 2_460 * M),
    ('2024-08-28', 24_530 * M), ('2024-11-20', 24_480 * M),    # first filing after the 10:1
]
NVDA_SPLITS = [(date(2021, 8, 20), 4), (date(2024, 8, 28), 10)]


def diluted_shares(*readings):
    """One period's diluted share count as reported by successive filings: (filing date, value) pairs."""
    return ('WeightedAverageNumberOfDilutedSharesOutstanding', 'shares',
            [fact('2024-01-01', '2024-12-31', v, filed=filed) for filed, v in readings])


def test_cutoff_is_the_first_filing_after_the_cover_page_count_jumps():
    assert find_splits(companyfacts(cover=NVDA_COVER)) == NVDA_SPLITS


def test_two_splits_years_apart_are_two_events():
    tesla = [('2020-07-28', 185 * M), ('2020-10-26', 930 * M), ('2022-07-25', 1_040 * M), ('2022-10-24', 3_100 * M)]
    assert find_splits(companyfacts(cover=tesla)) == [(date(2020, 10, 26), 5), (date(2022, 10, 24), 3)]


def test_split_is_found_without_a_ratio_tag_and_a_late_tag_is_ignored():
    # Walmart's 3:1: its ratio was only tagged in June, three months after the first post-split filing
    late_tag = {'end': '2024-02-26', 'val': 3, 'accn': 'x', 'form': '8-K', 'filed': '2024-06-07'}
    walmart = companyfacts(('StockholdersEquityNoteStockSplitConversionRatio1', 'pure', [late_tag]),
                           cover=[('2023-11-30', 2_690 * M), ('2024-03-15', 8_050 * M)])
    assert find_splits(walmart) == [(date(2024, 3, 15), 3)]


def test_reverse_split_is_detected_with_a_ratio_below_one():
    assert find_splits(companyfacts(cover=[('2011-05-05', 29_000 * M), ('2011-08-05', 2_900 * M)])) == \
        [(date(2011, 8, 5), 0.1)]


def test_ordinary_share_count_changes_are_not_splits():
    # buybacks, issuance, and an acquisition adding 67% (Capital One / Discover) are not split-like
    cover = [('2025-01-01', 380 * M), ('2025-02-01', 372 * M), ('2025-05-07', 383 * M), ('2025-07-31', 640 * M)]
    assert find_splits(companyfacts(cover=cover)) == []


def test_zero_placeholder_counts_are_ignored():
    assert find_splits(companyfacts(cover=[('2022-02-11', 0), ('2022-04-28', 50 * M), ('2022-07-27', 51 * M)])) == []


def test_acquisition_that_doubles_the_share_count_is_rejected():
    # the same period read before and after the jump is unchanged, so nothing was restated: not a split
    cf = companyfacts(diluted_shares(('2025-02-01', 100 * M), ('2025-05-01', 100 * M)),
                      cover=[('2025-02-01', 100 * M), ('2025-05-01', 200 * M)])
    assert find_splits(cf) == []


def test_split_that_restates_an_old_period_is_accepted():
    cf = companyfacts(diluted_shares(('2025-02-01', 100 * M), ('2025-05-01', 1_000 * M)),
                      cover=[('2025-02-01', 100 * M), ('2025-05-01', 1_000 * M)])
    assert find_splits(cf) == [(date(2025, 5, 1), 10)]


def test_split_with_no_restated_period_yet_is_accepted():
    cf = companyfacts(cover=[('2025-02-01', 100 * M), ('2025-05-01', 1_000 * M)])
    assert find_splits(cf) == [(date(2025, 5, 1), 10)]


def test_overrides_replace_detection(monkeypatch):
    monkeypatch.setitem(xbrl.SPLIT_OVERRIDES, 'XYZ', [(date(2022, 7, 26), 20)])
    assert find_splits(companyfacts(cover=NVDA_COVER), 'XYZ') == [(date(2022, 7, 26), 20)]


def test_split_factor_applies_only_to_values_filed_before_the_cutoff():
    assert split_factor('2024-08-27', NVDA_SPLITS) == 10       # day before: pre-split basis (only the 10:1 is after)
    assert split_factor('2024-08-28', NVDA_SPLITS) == 1        # the cutoff filing itself is post-split
    assert split_factor('2021-02-18', NVDA_SPLITS) == 40       # before both splits compounds
    assert split_factor('2022-01-01', NVDA_SPLITS) == 10


def nvda_eps(*rows):
    return companyfacts(('EarningsPerShareDiluted', 'USD/shares', list(rows)), cover=NVDA_COVER)


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


def test_split_adjusted_eps_keeps_the_precision_of_the_pre_split_filing():
    rows = by_period(normalize(nvda_eps(
        fact('2022-01-31', '2023-01-29', 1.74, filed='2024-02-21'),   # pre-split: 1.74 / 10 = 0.174
        fact('2022-01-31', '2023-01-29', 0.17, filed='2025-02-26'),   # same figure, rounded after the split
    ), 'NVDA'), 'eps_diluted')
    assert rows[(2023, 'FY')]['value'] == 0.174
    assert rows[(2023, 'FY')]['split_factor'] == 10
    assert rows[(2023, 'FY')]['filed'] == '2024-02-21'   # provenance points at the filing the value came from


def test_a_restated_eps_is_not_replaced_by_the_older_more_precise_reading():
    rows = by_period(normalize(nvda_eps(
        fact('2022-01-31', '2023-01-29', 1.74, filed='2024-02-21'),
        fact('2022-01-31', '2023-01-29', 0.20, filed='2025-02-26'),   # restated: 0.174 is 0.026 away, beyond rounding
    ), 'NVDA'), 'eps_diluted')
    assert rows[(2023, 'FY')]['value'] == 0.20
    assert rows[(2023, 'FY')]['split_factor'] == 1


def test_eps_without_a_split_between_readings_uses_the_latest():
    rows = by_period(normalize(companyfacts(('EarningsPerShareDiluted', 'USD/shares', [
        fact('2023-01-01', '2023-12-31', 11.95, filed='2024-02-23'),
        fact('2023-01-01', '2023-12-31', 11.95, filed='2025-02-20')])), 'COF'), 'eps_diluted')
    assert rows[(2023, 'FY')]['value'] == 11.95
    assert rows[(2023, 'FY')]['filed'] == '2025-02-20'


def test_non_per_share_metrics_are_never_split_adjusted():
    rows = normalize(companyfacts(('NetIncomeLoss', 'USD', [
        fact('2021-01-31', '2022-01-30', 9_752 * M, filed='2022-03-18'),
        fact('2022-01-31', '2023-01-29', 4_368 * M, filed='2023-02-24')]), cover=NVDA_COVER), 'NVDA')
    out = by_period(rows, 'net_income')
    assert out[(2022, 'FY')]['value'] == 9_752 * M
    assert out[(2022, 'FY')]['split_factor'] == 1


def test_company_without_splits_is_unadjusted():
    rows = by_period(normalize(companyfacts(('EarningsPerShareDiluted', 'USD/shares', [
        fact('2023-01-01', '2023-12-31', 12.0, filed='2024-02-20'),
        fact('2024-01-01', '2024-12-31', 14.0, filed='2025-02-20')])), 'COF'), 'eps_diluted')
    assert [rows[(y, 'FY')]['split_factor'] for y in (2023, 2024)] == [1, 1]


# --- consistency check -------------------------------------------------------------------------------------------

# NVIDIA FY2022 EPS as read in two filings: post-4:1 / pre-10:1, then restated after the 10:1 (0.385 rounds to 0.39)
EPS_FY2022 = [fact('2021-01-31', '2022-01-30', 3.85, filed='2024-02-21'),
              fact('2021-01-31', '2022-01-30', 0.39, filed='2025-02-26')]


def test_consistency_check_passes_when_the_cutoffs_are_right():
    cf = companyfacts(('EarningsPerShareDiluted', 'USD/shares', EPS_FY2022), cover=NVDA_COVER)
    assert check_split_consistency(cf, 'NVDA', NVDA_SPLITS) == (1, [])


def test_consistency_check_flags_a_cutoff_that_is_one_filing_late():
    cf = companyfacts(('EarningsPerShareDiluted', 'USD/shares', EPS_FY2022), cover=NVDA_COVER)
    late = [(date(2021, 8, 20), 4), (date(2025, 3, 1), 10)]   # the 10:1 cutoff lands after the restated filing
    checked, bad = check_split_consistency(cf, 'NVDA', late)
    assert checked == 1 and len(bad) == 1


def test_periods_with_a_single_reading_are_not_checked():
    cf = companyfacts(('EarningsPerShareDiluted', 'USD/shares',
                       [fact('2021-01-31', '2022-01-30', 3.85, filed='2024-02-21')]), cover=NVDA_COVER)
    assert check_split_consistency(cf, 'NVDA', NVDA_SPLITS) == (0, [])


def test_a_wrong_cutoff_is_classified_as_split_sized():
    cf = companyfacts(('EarningsPerShareDiluted', 'USD/shares', EPS_FY2022), cover=NVDA_COVER)
    late = [(date(2021, 8, 20), 4), (date(2025, 3, 1), 10)]
    assert check_split_consistency(cf, 'NVDA', late)[1][0]['kind'] == 'split'


def test_a_genuine_restatement_is_reported_but_not_split_sized():
    # Tesla restated Q1 2024 EPS from 0.34 to 0.41 for an accounting change; that must not look like a bad cutoff
    cf = companyfacts(('EarningsPerShareDiluted', 'USD/shares', [
        fact('2024-01-01', '2024-03-31', 0.34, filed='2024-04-24', form='10-Q'),
        fact('2024-01-01', '2024-03-31', 0.41, filed='2025-04-23', form='10-Q')]))
    checked, bad = check_split_consistency(cf, 'NVDA', [])
    assert checked == 1 and [b['kind'] for b in bad] == ['restated']


def test_reverse_split_rounding_is_not_a_mismatch():
    # Citigroup 1:10 reverse split: -0.80 as originally reported becomes -8.0, against a restated -7.99.
    # Scaling by 10 also scales the original reading's 0.005 rounding error to 0.05.
    cf = companyfacts(('EarningsPerShareDiluted', 'USD/shares', [
        fact('2021-01-01', '2021-12-31', -0.8, filed='2023-02-25'),
        fact('2021-01-01', '2021-12-31', -7.99, filed='2024-02-24')]))
    assert check_split_consistency(cf, 'NVDA', [(date(2023, 8, 5), 0.1)]) == (1, [])


# --- ingestion safety net ----------------------------------------------------------------------------------------

def with_core_metrics(cf):
    """The metrics resolve() insists on (every company has them), alongside whatever a test is about."""
    year = fact('2021-01-31', '2022-01-30', 1)
    for concept in ('Revenues', 'NetIncomeLoss', 'NetCashProvidedByUsedInOperatingActivities'):
        cf['facts']['us-gaap'][concept] = {'units': {'USD': [year]}}
    return cf


def stub_ingest(monkeypatch, cf):
    """Run ingest_facts for one ticker against canned facts and a fake store; returns what would be stored."""
    stored = []
    monkeypatch.setattr(xbrl.settings, 'tickers', ['NVDA'])
    monkeypatch.setattr(xbrl, 'fetch_companyfacts', lambda ticker, refresh=False: with_core_metrics(cf))
    monkeypatch.setattr('earnings_rag.store.init_schema', lambda: None)
    monkeypatch.setattr('earnings_rag.store.upsert_facts', stored.append)
    monkeypatch.setattr(xbrl, 'attach_income_statement', lambda facts, ticker, refresh=False: False)  # network
    monkeypatch.setattr('earnings_rag.segments.ingest_segments', lambda ticker, cf: True)  # reads data/raw, not hermetic
    return stored


def test_ingest_refuses_to_store_a_ticker_whose_split_check_fails(monkeypatch):
    # a split nothing detected (a multi-class issuer): the same period reads 20x apart in two filings
    cf = companyfacts(('EarningsPerShareDiluted', 'USD/shares', [
        fact('2021-01-01', '2021-12-31', 58.0, filed='2022-02-02'),
        fact('2021-01-01', '2021-12-31', 2.9, filed='2023-02-03')]))
    stored = stub_ingest(monkeypatch, cf)
    with pytest.raises(SystemExit, match='NVDA'):
        xbrl.ingest_facts()
    assert stored == []


def test_ingest_stores_a_ticker_whose_split_check_passes(monkeypatch):
    stored = stub_ingest(monkeypatch, companyfacts(('EarningsPerShareDiluted', 'USD/shares', EPS_FY2022),
                                                   cover=NVDA_COVER))
    xbrl.ingest_facts()
    assert len(stored) == 1 and stored[0]


# --- balances (values on a date) ----------------------------------------------------------------------------------

def instant(end, val, filed, form='10-Q', accn='b'):
    return {'end': end, 'val': val, 'accn': accn, 'form': form, 'filed': filed}


def cof_deposits_facts(*rows):
    flow = [fact('2023-01-01', '2023-12-31', 1), fact('2024-01-01', '2024-12-31', 2)]
    return companyfacts(('Revenues', 'USD', flow), ('Deposits', 'USD', list(rows)))


def test_balance_is_labelled_by_its_date_and_never_derived():
    rows = normalize(cof_deposits_facts(
        instant('2024-03-31', 351, '2024-05-01'), instant('2024-09-30', 353, '2024-11-01'),
        instant('2024-12-31', 362, '2025-02-20', form='10-K'),
        instant('2024-12-31', 362, '2025-05-01'),          # repeated as a comparison in a later 10-Q
        instant('2024-05-15', 999, '2024-08-01')), 'COF')   # not a quarter end
    out = by_period(rows, 'deposits')
    assert {k: r['value'] for k, r in out.items()} == {(2024, 'Q1'): 351, (2024, 'Q3'): 353, (2024, 'FY'): 362}
    assert not any(r['derived'] for r in out.values())          # no Q2, no Q4: balances cannot be subtracted
    assert out[(2024, 'FY')]['period_start'] == out[(2024, 'FY')]['period_end'] == '2024-12-31'


def test_period_after_the_last_annual_fact_gets_its_balance_in_the_year_in_progress():
    out = by_period(normalize(cof_deposits_facts(instant('2025-03-31', 367, '2025-05-01')), 'COF'), 'deposits')
    assert out[(2025, 'Q1')]['value'] == 367
