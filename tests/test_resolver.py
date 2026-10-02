import json

import pytest

from earnings_rag import xbrl
from earnings_rag.config import settings
from earnings_rag.resolver import resolve, describe, METRICS, CORE

M = 1_000_000


def annual(val, end='2024-12-31', start='2024-01-01', filed='2025-02-01'):
    return {'start': start, 'end': end, 'val': val * M, 'accn': 'a', 'form': '10-K', 'filed': filed}


def facts(**concepts):
    """companyfacts for one fiscal year (2024) with the given {concept: millions}; the core metrics are filled in."""
    base = {'Revenues': 100, 'NetIncomeLoss': 10, 'NetCashProvidedByUsedInOperatingActivities': 12}
    gaap = {c: {'units': {'USD': [annual(v)]}} for c, v in {**base, **concepts}.items() if v is not None}
    gaap['EarningsPerShareDiluted'] = {'units': {'USD/shares': [{**annual(1), 'val': 1.0}]}}
    return {'facts': {'us-gaap': gaap}}


def test_synonyms_that_agree_are_taken_as_is():
    res = resolve(facts(RevenueFromContractWithCustomerExcludingAssessedTax=100), 'XYZ')
    assert res.chosen['revenue'] == ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax']
    assert 'revenue' not in res.skipped


def test_disagreeing_synonyms_are_accepted_when_an_identity_confirms_the_largest():
    cf = facts(RevenueFromContractWithCustomerExcludingAssessedTax=70, CostOfRevenue=60, GrossProfit=40)
    res = resolve(cf, 'XYZ')
    assert res.primary['revenue'] == 'Revenues'                      # 100 - 60 = 40 = gross profit
    assert ('revenue: revenue - cost = gross profit', True) in res.checks


def test_disagreeing_synonyms_without_a_confirming_identity_are_skipped_with_the_reason():
    res = resolve(facts(RevenueFromContractWithCustomerExcludingAssessedTax=70), 'XYZ')
    assert 'revenue' not in res.chosen and 'revenue' in res.missing_core
    assert 'Revenues=100M' in res.skipped['revenue'] and '=70M' in res.skipped['revenue']
    assert any('SKIPPED revenue' in line for line in describe('XYZ', res))


def test_largest_cost_synonym_needs_an_identity_too():
    # Caterpillar: CostOfRevenue 44,752M next to a stray CostOfGoodsAndServicesSold of 49M, nothing to confirm either
    res = resolve(facts(CostOfRevenue=60, CostOfGoodsAndServicesSold=1), 'XYZ')
    assert 'cost_of_revenue' in res.skipped
    ok = resolve(facts(CostOfRevenue=60, CostOfGoodsAndServicesSold=1, GrossProfit=40), 'XYZ')
    assert ok.primary['cost_of_revenue'] == 'CostOfRevenue'


def test_revenue_is_confirmed_by_total_costs_and_by_the_bank_identity():
    assert resolve(facts(RevenueFromContractWithCustomerExcludingAssessedTax=70, CostsAndExpenses=75,
                         OperatingIncomeLoss=25), 'XYZ').primary['revenue'] == 'Revenues'
    bank = resolve(facts(RevenueFromContractWithCustomerExcludingAssessedTax=15, InterestIncomeExpenseNet=80,
                         NoninterestIncome=20), 'XYZ')
    assert bank.primary['revenue'] == 'Revenues' and bank.is_bank


def test_bank_metrics_resolve_only_for_a_bank():
    # a manufacturer with a small captive finance arm also tags a loan-loss provision; it is not a bank metric for it
    industrial = resolve(facts(ProvisionForLoanLossesExpensed=1), 'XYZ')
    assert not industrial.is_bank and 'provision_for_credit_losses' in industrial.absent
    bank = resolve(facts(ProvisionForLoanLossesExpensed=5, InterestIncomeExpenseNet=80, NoninterestIncome=20), 'XYZ')
    assert bank.is_bank and 'provision_for_credit_losses' in bank.chosen


def test_an_override_pins_a_concept_and_is_reported():
    cf = facts(InterestIncomeExpenseNet=80, NoninterestIncome=20, Deposits=300, NotesReceivableGross=250)
    res = resolve(cf, 'COF')
    assert res.primary['loans'] == 'NotesReceivableGross' and ('deposits', 'Deposits') in res.pinned
    assert 'loans' not in resolve(cf, 'XYZ').chosen                 # no override for another bank: not guessed
    assert any(line.startswith('  override: loans') for line in describe('COF', res))


def test_missing_core_metric_is_named_and_a_concept_that_fits_the_identity_is_suggested():
    # revenue under a name nobody listed: X - cost = gross profit finds it
    cf = facts(Revenues=None, GrossProfit=40, CostOfRevenue=60, TotalNetSalesCustomName=100)
    res = resolve(cf, 'XYZ')
    assert res.missing_core == ['revenue']
    assert any('TotalNetSalesCustomName' in s for s in res.suggestions['revenue'])
    assert any('MISSING CORE METRIC revenue' in line and 'TotalNetSalesCustomName' in line
               for line in describe('XYZ', res))


def test_a_ticker_without_annual_data_is_reported_not_crashed_on():
    res = resolve({'facts': {'us-gaap': {'NetIncomeLoss': {'units': {'USD': []}}}}}, 'XOM')
    assert res.no_data and 'no annual data' in describe('XOM', res)[0]


def test_ingest_refuses_a_ticker_with_a_missing_core_metric(monkeypatch, capsys):
    cf = facts(Revenues=None)
    monkeypatch.setattr(xbrl.settings, 'tickers', ['XYZ'])
    monkeypatch.setattr(xbrl, 'fetch_companyfacts', lambda ticker, refresh=False: cf)
    monkeypatch.setattr('earnings_rag.store.init_schema', lambda: None)
    monkeypatch.setattr(xbrl, 'attach_income_statement', lambda facts, ticker, refresh=False: False)  # network
    stored = []
    monkeypatch.setattr('earnings_rag.store.upsert_facts', stored.append)
    with pytest.raises(SystemExit):
        xbrl.ingest_facts()
    assert stored == [] and 'MISSING CORE METRIC revenue' in capsys.readouterr().out


def test_every_metric_has_what_the_resolver_needs():
    assert set(CORE) <= set(METRICS)
    for metric, spec in METRICS.items():
        assert spec['unit'] and (spec['synonyms'] or spec.get('group') == 'bank'), metric


PRETAX = 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'
CASH = 'NetCashProvidedByUsedInOperatingActivities'

# What a person verified by hand before the resolver existed, for the three companies in the corpus. The cached SEC
# files are git-ignored, so CI skips this; run it locally after `python -m earnings_rag.xbrl`.
HAND_VERIFIED = {
    'NVDA': {'revenue': 'Revenues', 'net_income': 'NetIncomeLoss', 'eps_diluted': 'EarningsPerShareDiluted',
             'operating_cash_flow': CASH, 'gross_profit': 'GrossProfit', 'operating_income': 'OperatingIncomeLoss',
             'rnd_expense': 'ResearchAndDevelopmentExpense', 'cost_of_revenue': 'CostOfRevenue',
             'pretax_income': PRETAX},
    'AAPL': {'revenue': 'RevenueFromContractWithCustomerExcludingAssessedTax', 'net_income': 'NetIncomeLoss',
             'eps_diluted': 'EarningsPerShareDiluted', 'operating_cash_flow': CASH, 'gross_profit': 'GrossProfit',
             'operating_income': 'OperatingIncomeLoss', 'rnd_expense': 'ResearchAndDevelopmentExpense',
             'cost_of_revenue': 'CostOfGoodsAndServicesSold', 'pretax_income': PRETAX},
    'COF': {'revenue': 'Revenues', 'net_income': 'NetIncomeLoss', 'eps_diluted': 'EarningsPerShareDiluted',
            'operating_cash_flow': CASH, 'pretax_income': PRETAX, 'net_interest_income': 'InterestIncomeExpenseNet',
            'noninterest_income': 'NoninterestIncome', 'noninterest_expense': 'NoninterestExpense',
            'provision_for_credit_losses': 'ProvisionForLoanLossesExpensed', 'loans': 'NotesReceivableGross',
            'deposits': 'Deposits'},
}


@pytest.mark.parametrize('ticker', sorted(HAND_VERIFIED))
def test_resolver_reproduces_the_hand_verified_map(ticker):
    path = settings.data_dir / 'xbrl' / f'{ticker}.json'
    if not path.exists():
        pytest.skip('SEC facts not cached (data/ is git-ignored)')
    res = resolve(json.loads(path.read_text(encoding='utf-8')), ticker)
    assert res.primary == HAND_VERIFIED[ticker] and not res.skipped and not res.missing_core
