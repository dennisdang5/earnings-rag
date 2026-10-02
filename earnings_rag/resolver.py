from dataclasses import dataclass, field
from datetime import date

# metric -> the XBRL concepts that mean it. Written once, for every company: a company-specific map does not scale to
# many tickers. Only true synonyms belong here; look-alikes are different numbers and must stay out ("pre-tax income,
# domestic" is only the US part, "interest and dividend income" is gross, not net, interest).
# additive=False for per-share values: share counts change, so FY EPS minus 9-month EPS is not Q4 EPS.
# per_share=True values are rescaled to today's share count (see xbrl.find_splits).
# profit=True: a profit figure; the tool reminds the model that its margin is the figure divided by revenue.
# balance=True: a value on a date (the period's last day), not an amount over the period. Never added or derived.
# group='bank': only resolved for a company that passes the bank test (is_bank); a non-bank can tag the same concept
# for a small captive finance arm (Caterpillar's ProvisionForLoanLossesExpensed is 109M on 67,589M revenue).
METRICS = {
    'revenue': {
        'synonyms': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax',
                     'RevenueFromContractWithCustomerIncludingAssessedTax', 'RevenuesNetOfInterestExpense',
                     'SalesRevenueNet'],
        'unit': 'USD', 'additive': True},
    'net_income': {'synonyms': ['NetIncomeLoss'], 'unit': 'USD', 'additive': True, 'profit': True},
    'eps_diluted': {'synonyms': ['EarningsPerShareDiluted'], 'unit': 'USD/shares', 'additive': False,
                    'per_share': True},
    'operating_cash_flow': {'synonyms': ['NetCashProvidedByUsedInOperatingActivities'], 'unit': 'USD',
                            'additive': True},
    'gross_profit': {'synonyms': ['GrossProfit'], 'unit': 'USD', 'additive': True, 'profit': True},
    'operating_income': {'synonyms': ['OperatingIncomeLoss'], 'unit': 'USD', 'additive': True, 'profit': True},
    'rnd_expense': {'synonyms': ['ResearchAndDevelopmentExpense',
                                 'ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost'],
                    'unit': 'USD', 'additive': True},
    'cost_of_revenue': {'synonyms': ['CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfGoodsSold'],
                        'unit': 'USD', 'additive': True},
    'pretax_income': {
        'synonyms': ['IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'],
        'unit': 'USD', 'additive': True, 'profit': True},
    'net_interest_income': {'synonyms': ['InterestIncomeExpenseNet'], 'unit': 'USD', 'additive': True,
                            'group': 'bank'},
    'noninterest_income': {'synonyms': ['NoninterestIncome'], 'unit': 'USD', 'additive': True, 'group': 'bank'},
    'noninterest_expense': {'synonyms': ['NoninterestExpense'], 'unit': 'USD', 'additive': True, 'group': 'bank'},
    'provision_for_credit_losses': {
        'synonyms': ['ProvisionForLoanLossesExpensed', 'ProvisionForLoanLeaseAndOtherLosses'],
        'unit': 'USD', 'additive': True, 'group': 'bank'},
    # no synonym list: no concept means "loans" for every bank, so each is pinned in OVERRIDES
    'loans': {'synonyms': [], 'unit': 'USD', 'additive': False, 'balance': True, 'group': 'bank'},
    'deposits': {'synonyms': [], 'unit': 'USD', 'additive': False, 'balance': True, 'group': 'bank'},
}

# (ticker, metric) -> concept, for what no rule can settle. Printed at ingest so a human sees every pin.
OVERRIDES = {
    ('COF', 'loans'): 'NotesReceivableGross',
    ('COF', 'deposits'): 'Deposits',
}

# Every company has these, so a missing one means a synonym is missing here, not a fact about the company.
CORE = ('revenue', 'net_income', 'eps_diluted', 'operating_cash_flow')

TOLERANCE = 1e6          # an identity holds if it is within a million dollars: the filings round to millions
AMBIGUITY = 0.001        # synonyms whose values differ by less than this (relative) are the same number


@dataclass
class Resolution:
    chosen: dict[str, list[str]] = field(default_factory=dict)   # metric -> synonym concepts present (largest wins per period)
    primary: dict[str, str] = field(default_factory=dict)        # metric -> the concept in use in the latest fiscal year
    skipped: dict[str, str] = field(default_factory=dict)        # metric -> why it was not stored
    absent: list[str] = field(default_factory=list)              # metrics the company does not report
    pinned: list[tuple[str, str]] = field(default_factory=list)  # overrides that applied
    checks: list[tuple[str, bool]] = field(default_factory=list)  # accounting identities, report-only
    suggestions: dict[str, list[str]] = field(default_factory=dict)
    missing_core: list[str] = field(default_factory=list)
    is_bank: bool = False
    no_data: bool = False


def _days(r: dict) -> int:
    return (date.fromisoformat(r['end']) - date.fromisoformat(r['start'])).days


def _rows(gaap: dict, concept: str, unit: str) -> list[dict]:
    return gaap.get(concept, {}).get('units', {}).get(unit, [])


def _latest_annual_end(gaap: dict) -> str | None:
    """The latest fiscal year end any flow metric has a full-year fact for."""
    ends = [r['end'] for spec in METRICS.values() if not spec.get('balance') for c in spec['synonyms']
            for r in _rows(gaap, c, spec['unit']) if 'start' in r and 350 <= _days(r) <= 380]
    return max(ends) if ends else None


def _value_at(gaap: dict, concept: str, unit: str, end: str, balance: bool = False) -> float | None:
    """The latest-filed reading of a full-year fact (or a balance) ending on `end`."""
    rows = [r for r in _rows(gaap, concept, unit) if r['end'] == end
            and (('start' not in r) if balance else ('start' in r and 350 <= _days(r) <= 380))]
    return max(rows, key=lambda r: r['filed'])['val'] if rows else None


def _close(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) <= TOLERANCE


def _fy_values(gaap: dict, end: str) -> dict[str, float | None]:
    """The other figures an identity needs, at the latest fiscal year end."""
    names = {'gp': 'GrossProfit', 'oi': 'OperatingIncomeLoss', 'costs_and_expenses': 'CostsAndExpenses',
             'opex': 'OperatingExpenses', 'sga': 'SellingGeneralAndAdministrativeExpense',
             'nii': 'InterestIncomeExpenseNet', 'nonii': 'NoninterestIncome', 'tax': 'IncomeTaxExpenseBenefit',
             'pretax': METRICS['pretax_income']['synonyms'][0], 'ni': 'NetIncomeLoss'}
    return {k: _value_at(gaap, c, 'USD', end) for k, c in names.items()}


def _revenue_identities(revenue: float, cost: float | None, v: dict) -> list[str]:
    """Which accounting identities a revenue figure satisfies. Any one confirms it; a smaller look-alike satisfies none."""
    found = []
    if v['costs_and_expenses'] and _close(revenue - v['costs_and_expenses'], v['oi']):
        found.append('revenue - costs and expenses = operating income')
    if cost is not None and v['opex'] and _close(revenue - cost - v['opex'], v['oi']):
        found.append('revenue - cost - operating expenses = operating income')
    if cost is not None and v['sga'] and _close(revenue - cost - v['sga'], v['oi']):
        found.append('revenue - cost - SG&A = operating income')
    if cost is not None and v['gp'] is not None and _close(revenue - cost, v['gp']):
        found.append('revenue - cost = gross profit')
    if v['nii'] is not None and v['nonii'] is not None and _close(v['nii'] + v['nonii'], revenue):
        found.append('net interest income + non-interest income = revenue')
    return found


def _cost_identities(cost: float, revenue: float | None, v: dict) -> list[str]:
    if revenue is None:
        return []
    found = []
    if v['gp'] is not None and _close(revenue - cost, v['gp']):
        found.append('revenue - cost = gross profit')
    if v['opex'] and _close(revenue - cost - v['opex'], v['oi']):
        found.append('revenue - cost - operating expenses = operating income')
    if v['sga'] and _close(revenue - cost - v['sga'], v['oi']):
        found.append('revenue - cost - SG&A = operating income')
    return found


def _suggest(gaap: dict, end: str, targets: list[float], tried: set[str]) -> list[str]:
    """Concepts X for which X - Y equals a reported profit figure for some Y: candidates for a missing top line."""
    values = {}
    for c in gaap:
        v = _value_at(gaap, c, 'USD', end)
        if v is not None and v > 0:
            values[c] = v
    hits = {}
    for x, vx in values.items():
        if x in tried:
            continue
        for target in targets:
            if vx > target * 1.05 and any(y != x and _close(vx - vy, target) for y, vy in values.items()):
                hits[x] = vx
    return [f'{c} ({v / 1e6:,.0f}M)' for c, v in sorted(hits.items(), key=lambda kv: -kv[1])[:5]]


def resolve(companyfacts: dict, ticker: str) -> Resolution:
    """
    Decide which XBRL concepts carry each metric for one company, from its own data. Pure: no network, no database.

    A metric with one synonym present, or several that agree, is taken as is. Several that disagree (Capital One has
    Revenues 53,434M and a fee-only revenue concept of 8,062M) are only accepted if the largest satisfies an accounting
    identity; an unverified one is not stored, so the agent says "not reported" instead of giving a wrong figure.
    """
    gaap = companyfacts['facts']['us-gaap']
    res = Resolution()
    end = _latest_annual_end(gaap)
    if end is None:
        res.no_data = True
        return res
    v = _fy_values(gaap, end)

    def candidates(metric: str) -> dict[str, float]:
        """synonym -> its latest-year value, for synonyms that have any data in any year."""
        spec = METRICS[metric]
        pin = OVERRIDES.get((ticker, metric))
        found = {}
        for c in ([pin] if pin else spec['synonyms']):
            if any(True for _ in _rows(gaap, c, spec['unit'])):
                found[c] = _value_at(gaap, c, spec['unit'], end, bool(spec.get('balance')))
        return found

    # the bank test: net interest income + non-interest income equals revenue for a bank, and for nothing else
    rev_values = [x for x in candidates('revenue').values() if x is not None]
    res.is_bank = any(_close(v['nii'] + v['nonii'], x) for x in rev_values if v['nii'] and v['nonii'])

    revenue = max(rev_values) if rev_values else None
    cost_values = [x for x in candidates('cost_of_revenue').values() if x is not None]
    cost = max(cost_values) if cost_values else None

    for metric, spec in METRICS.items():
        pinned = (ticker, metric) in OVERRIDES
        if spec.get('group') == 'bank' and not res.is_bank and not pinned:
            res.absent.append(metric)
            continue
        found = candidates(metric)
        if not found:
            res.absent.append(metric)
            continue

        current = {c: x for c, x in found.items() if x is not None}
        top = max(current.values(), default=None)
        disagree = top is not None and any(abs(x - top) > AMBIGUITY * abs(top) for x in current.values())
        if disagree and not pinned:
            if metric == 'revenue':
                confirmed = _revenue_identities(top, cost, v)
            elif metric == 'cost_of_revenue':
                confirmed = _cost_identities(top, revenue, v)
            else:
                confirmed = []
            if not confirmed:
                res.skipped[metric] = ('synonyms disagree and no identity confirms the largest: '
                                       + ', '.join(f'{c}={x / 1e6:,.0f}M' for c, x in current.items()))
                continue
            res.checks.append((f'{metric}: {confirmed[0]}', True))

        res.chosen[metric] = list(found)
        res.primary[metric] = (max(current, key=current.get) if current else next(iter(found)))
        if pinned:
            res.pinned.append((metric, res.primary[metric]))

    # report-only identities that did not decide anything: a flag for a human, not a gate
    if v['gp'] is not None and revenue is not None and cost is not None:
        res.checks.append(('gross profit = revenue - cost', _close(revenue - cost, v['gp'])))
    if v['pretax'] is not None and v['tax'] is not None and v['ni'] is not None:
        res.checks.append(('pre-tax income - tax ~ net income (1%)',
                           abs(v['pretax'] - v['tax'] - v['ni']) <= 0.01 * abs(v['ni'])))

    res.missing_core = [m for m in CORE if m not in res.chosen]
    if 'revenue' in res.missing_core:
        targets = [t for t in (v['oi'], v['gp']) if t]
        tried = set(METRICS['revenue']['synonyms'])
        res.suggestions['revenue'] = _suggest(gaap, end, targets, tried) if targets else []
    return res


def describe(ticker: str, res: Resolution) -> list[str]:
    """The coverage report printed at ingest: what was found, what was not, what was pinned, what needs a look."""
    if res.no_data:
        return [f'{ticker}: no annual data in the SEC facts (wrong SEC entity for this ticker?)']
    lines = [f'{ticker}: metrics found {len(res.chosen)}/{len(METRICS)}: {", ".join(res.chosen)}'
             + (' (bank)' if res.is_bank else '')]
    if res.absent:
        lines.append(f'  not reported: {", ".join(res.absent)}')
    for metric, why in res.skipped.items():
        lines.append(f'  SKIPPED {metric}: {why}')
    for metric, concept in res.pinned:
        lines.append(f'  override: {metric} = {concept}')
    for label, ok in res.checks:
        lines.append(f'  {"ok" if ok else "DIFFERS"}: {label}')
    for metric in res.missing_core:
        hint = res.suggestions.get(metric)
        lines.append(f'  MISSING CORE METRIC {metric}'
                     + (f'; concepts that fit the identity: {"; ".join(hint)}' if hint else ''))
    return lines
