from earnings_rag import statements, xbrl
from earnings_rag.resolver import resolve, describe
from earnings_rag.statements import parse_calc, income_statement, positions, cost_group, concepts_on

M = 1_000_000


def calc_link(role, arcs):
    """One calculationLink block. arcs: [(parent, child, weight)]; 'us-gaap_' is added unless a prefix is given."""
    def full(name):
        return name if '_' in name else f'us-gaap_{name}'
    labels = {}
    for parent, child, _ in arcs:
        for name in (parent, child):
            labels.setdefault(name, f'L{len(labels)}')
    locs = ''.join(f'<link:loc xlink:href="x.xsd#{full(n)}" xlink:label="{lab}"/>' for n, lab in labels.items())
    arc_xml = ''.join(f'<link:calculationArc xlink:from="{labels[p]}" xlink:to="{labels[c]}" weight="{w}"/>'
                      for p, c, w in arcs)
    return f'<link:calculationLink xlink:role="http://x/role/{role}">{locs}{arc_xml}</link:calculationLink>'


def linkbase(*links):
    return ('<link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase" xmlns:xlink="http://www.w3.org/1999/xlink">'
            + ''.join(links) + '</link:linkbase>')


def calc_xml(role, arcs):
    return linkbase(calc_link(role, arcs))


# NVIDIA: gross profit on the statement
NVDA_ARCS = [('NetIncomeLoss', 'IncomeTaxExpenseBenefit', -1),
             ('NetIncomeLoss', 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 1),
             ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
              'OperatingIncomeLoss', 1),
             ('OperatingIncomeLoss', 'GrossProfit', 1), ('OperatingIncomeLoss', 'OperatingExpenses', -1),
             ('GrossProfit', 'Revenues', 1), ('GrossProfit', 'CostOfRevenue', -1)]

# UnitedHealth: no gross profit, costs and expenses beside revenue, a small cost of goods sold among them
UNH_ARCS = [('NetIncomeLoss', 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 1),
            ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
             'OperatingIncomeLoss', 1),
            ('OperatingIncomeLoss', 'Revenues', 1), ('OperatingIncomeLoss', 'CostsAndExpenses', -1),
            ('CostsAndExpenses', 'PolicyholderBenefitsAndClaimsIncurredNet', 1),
            ('CostsAndExpenses', 'CostOfGoodsAndServicesSold', 1)]


def statement(arcs, role='ConsolidatedStatementsofIncome'):
    return income_statement(parse_calc(calc_xml(role, arcs)))


def test_income_statement_is_found_by_role_name_and_net_income_on_top():
    roles = parse_calc(linkbase(
        calc_link('ConsolidatedStatementsofComprehensiveIncome', [('ComprehensiveIncomeNetOfTax', 'NetIncomeLoss', 1)]),
        calc_link('IncomeTaxesDetails', [('NetIncomeLoss', 'IncomeTaxExpenseBenefit', -1)]),
        calc_link('ConsolidatedStatementsofIncome', NVDA_ARCS)))
    tree = income_statement(roles)
    assert tree is not None and 'GrossProfit' in tree
    assert income_statement(parse_calc(calc_xml('BalanceSheets', [('Assets', 'Cash', 1)]))) is None


def test_a_role_split_over_several_blocks_is_merged():
    tree = parse_calc(linkbase(calc_link('ConsolidatedStatementsofIncome', NVDA_ARCS[:3]),
                               calc_link('ConsolidatedStatementsofIncome', NVDA_ARCS[3:])))['ConsolidatedStatementsofIncome']
    assert 'GrossProfit' in tree and 'NetIncomeLoss' in tree


def test_company_concepts_keep_their_prefix():
    tree = statement([('NetIncomeLoss', 'bac_Provision', -1)])
    assert tree['NetIncomeLoss'] == [(-1.0, 'bac:Provision')]


def test_positions_under_gross_profit_and_under_operating_income():
    assert positions(statement(NVDA_ARCS)) == {
        'pretax_income': 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'cost_of_revenue': 'CostOfRevenue', 'revenue': 'Revenues'}
    unh = positions(statement(UNH_ARCS))
    assert unh['revenue'] == 'Revenues' and 'cost_of_revenue' not in unh


def test_bank_positions_and_a_branch_with_two_candidates_is_left_out():
    bank = [('NetIncomeLoss', 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 1),
            ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
             'RevenuesNetOfInterestExpense', 1),
            ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
             'ProvisionForLoanLeaseAndOtherLosses', -1),
            ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
             'NoninterestExpense', -1),
            ('RevenuesNetOfInterestExpense', 'InterestIncomeExpenseNet', 1)]
    got = positions(statement(bank))
    assert got['revenue'] == 'RevenuesNetOfInterestExpense' and got['provision_for_credit_losses'] == 'ProvisionForLoanLeaseAndOtherLosses'
    two = bank + [('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                   'OtherProvision', -1)]
    assert 'provision_for_credit_losses' not in positions(statement(two))


def test_cost_group_is_none_under_gross_profit_and_the_siblings_otherwise():
    assert cost_group(statement(NVDA_ARCS), 'CostOfRevenue') is None
    assert cost_group(statement(UNH_ARCS), 'CostOfGoodsAndServicesSold') == [
        'PolicyholderBenefitsAndClaimsIncurredNet', 'CostOfGoodsAndServicesSold']
    assert cost_group(statement(UNH_ARCS), 'NotOnTheStatement') == []


# --- the tree as a second opinion on resolve() ---------------------------------------------------------------------

def annual(val, end='2024-12-31', start='2024-01-01', form='10-K'):
    return {'start': start, 'end': end, 'val': val * M, 'accn': 'a', 'form': form, 'filed': '2025-02-01'}


def facts(tree=None, **concepts):
    base = {'Revenues': 100, 'NetIncomeLoss': 10, 'NetCashProvidedByUsedInOperatingActivities': 12}
    gaap = {c: {'units': {'USD': [annual(v)] if not isinstance(v, list) else v}}
            for c, v in {**base, **concepts}.items() if v is not None}
    gaap['EarningsPerShareDiluted'] = {'units': {'USD/shares': [{**annual(1), 'val': 1.0}]}}
    out = {'facts': {'us-gaap': gaap}}
    if tree:
        out['income_statement'] = tree
    return out


def test_cost_of_revenue_beside_a_larger_cost_is_rejected_but_the_main_cost_is_kept():
    unh = facts(statement(UNH_ARCS), CostOfGoodsAndServicesSold=50, PolicyholderBenefitsAndClaimsIncurredNet=314)
    res = resolve(unh, 'UNH')
    assert 'cost_of_revenue' not in res.chosen and 'not the main cost' in res.skipped['cost_of_revenue']
    assert 'PolicyholderBenefitsAndClaimsIncurredNet is 314M' in res.skipped['cost_of_revenue']
    amazon = facts(statement(UNH_ARCS), CostOfGoodsAndServicesSold=356, PolicyholderBenefitsAndClaimsIncurredNet=80)
    assert resolve(amazon, 'AMZN').primary['cost_of_revenue'] == 'CostOfGoodsAndServicesSold'
    # with no tree nothing can tell the two apart, which is the limitation this closes
    assert 'cost_of_revenue' in resolve(facts(CostOfGoodsAndServicesSold=50), 'UNH').chosen


def test_when_exactly_one_competing_concept_is_on_the_statement_it_settles_the_disagreement():
    # Caterpillar: CostOfRevenue is a line of the statement, a stray CostOfGoodsAndServicesSold is not
    tree = statement([('NetIncomeLoss', 'OperatingIncomeLoss', 1), ('OperatingIncomeLoss', 'Revenues', 1),
                      ('OperatingIncomeLoss', 'CostsAndExpenses', -1), ('CostsAndExpenses', 'CostOfRevenue', 1)])
    res = resolve(facts(tree, CostOfRevenue=60, CostOfGoodsAndServicesSold=1), 'CAT')
    assert res.primary['cost_of_revenue'] == 'CostOfRevenue'
    assert ('cost_of_revenue: only CostOfRevenue is on the income statement', True) in res.checks
    assert 'cost_of_revenue' in resolve(facts(None, CostOfRevenue=60, CostOfGoodsAndServicesSold=1), 'CAT').skipped


def test_a_metric_no_synonym_matched_is_taken_from_the_statement_when_the_concept_is_standard():
    variant = 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments'
    tree = statement([('NetIncomeLoss', variant, 1), (variant, 'OperatingIncomeLoss', 1)])
    res = resolve(facts(tree, **{variant: 12}), 'AMZN')
    assert res.primary['pretax_income'] == variant
    assert 'pretax_income' not in resolve(facts(None, **{variant: 12}), 'AMZN').chosen


def test_the_same_figure_under_another_name_is_agreement_not_a_difference():
    tree = statement([('NetIncomeLoss', 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 1),
                      ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                       'OperatingIncomeLoss', 1), ('OperatingIncomeLoss', 'RevenueFromContractWithCustomerExcludingAssessedTax', 1)])
    res = resolve(facts(tree, RevenueFromContractWithCustomerExcludingAssessedTax=100), 'TSLA')
    assert all(ok for label, ok in res.checks if 'revenue' in label)
    # a statement whose revenue line is a concept the resolver does not know, with a different figure: flagged, not applied
    other = statement([('NetIncomeLoss', 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 1),
                       ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                        'OperatingIncomeLoss', 1), ('OperatingIncomeLoss', 'SalesRevenueGoodsNet', 1)])
    different = resolve(facts(other, SalesRevenueGoodsNet=70), 'TSLA')
    assert different.primary['revenue'] == 'Revenues'
    assert any(not ok and 'income statement has SalesRevenueGoodsNet' in label for label, ok in different.checks)


def test_a_metric_two_years_behind_is_not_stored_and_one_year_is_only_flagged():
    old = [annual(5, '2021-12-31', '2021-01-01'), annual(5, '2022-12-31', '2022-01-01')]
    gp_old = facts(CostOfRevenue=60, GrossProfit=old, Revenues=[annual(100, '2023-12-31', '2023-01-01'), annual(100)])
    res = resolve(gp_old, 'XYZ')
    assert 'gross_profit' in res.discontinued and 'gross_profit' not in res.chosen
    assert any('STALE gross_profit' in line and 'not stored' in line for line in describe('XYZ', res))
    late = facts(GrossProfit=[annual(5, '2023-12-31', '2023-01-01')], CostOfRevenue=60)
    res = resolve(late, 'XYZ')
    assert res.stale['gross_profit'] == 2023 and 'gross_profit' in res.chosen


def test_a_stale_metric_whose_concept_moved_to_a_company_concept_says_so():
    tree = statement([('NetIncomeLoss', 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 1),
                      ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                       'RevenuesNetOfInterestExpense', 1),
                      ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                       'bac_ProvisionReversal', -1)])
    old = [annual(5, '2020-12-31', '2020-01-01'), annual(5, '2021-12-31', '2021-01-01')]
    res = resolve(facts(tree, InterestIncomeExpenseNet=80, NoninterestIncome=20, ProvisionForLoanLossesExpensed=old,
                        RevenuesNetOfInterestExpense=100), 'BAC')
    assert 'provision_for_credit_losses' in res.discontinued
    assert any('bac:ProvisionReversal' in note and 'not in the SEC facts' in note for note in res.notes)


def test_ten_qs_do_not_define_the_latest_fiscal_year():
    # Amazon's 10-Qs carry trailing-twelve-month figures that look like a fiscal year ending a quarter later
    ttm = {**annual(110, '2025-03-31', '2024-04-01'), 'form': '10-Q'}
    res = resolve(facts(Revenues=[annual(100), ttm]), 'AMZN')
    assert res.latest_fy == 2024 and not res.stale


def test_attach_income_statement_survives_a_download_failure(monkeypatch, capsys):
    def boom(ticker, facts, refresh=False):
        raise RuntimeError('SEC unavailable')
    monkeypatch.setattr(statements, 'fetch_calc', boom)
    cf = facts()
    assert xbrl.attach_income_statement(cf, 'XYZ') is False
    assert 'income_statement' not in cf and 'resolving without the income statement' in capsys.readouterr().out


def test_attach_income_statement_attaches_the_parsed_tree(monkeypatch):
    monkeypatch.setattr(statements, 'fetch_calc', lambda ticker, facts, refresh=False: calc_xml('Income', NVDA_ARCS))
    cf = facts()
    assert xbrl.attach_income_statement(cf, 'NVDA') is True
    assert 'GrossProfit' in concepts_on(cf['income_statement'])
