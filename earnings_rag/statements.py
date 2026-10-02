import time

from bs4 import BeautifulSoup

from earnings_rag.config import settings

# The calculation linkbase of a 10-K states the company's own arithmetic: NetIncomeLoss = + pre-tax - tax, pre-tax =
# + OperatingIncomeLoss + non-operating, and so on. It is used as a second opinion on resolver.resolve(): it confirms
# choices, settles some ambiguities and fills gaps, but is not the primary method because trees can be incomplete
# (Capital One's leaves pre-tax income without children and does not contain total revenue at all).

TOPS = ('NetIncomeLoss', 'ProfitLoss')
SUBTOTALS = {'GrossProfit', 'OperatingIncomeLoss'}


def fetch_calc(ticker: str, companyfacts: dict, refresh: bool = False) -> str | None:
    """The calculation linkbase of the company's latest 10-K, cached next to the SEC facts. None if there is none."""
    path = settings.data_dir / 'xbrl' / f'{ticker}_cal.xml'
    if path.exists() and not refresh:
        return path.read_text(encoding='utf-8')

    from earnings_rag.ingest import session  # raises at import without SEC_USER_AGENT, so not at the top

    rows = [r for d in companyfacts['facts']['us-gaap'].values() for unit in d.get('units', {}).values()
            for r in unit if r.get('form') == '10-K']
    if not rows:
        return None
    accn = max(rows, key=lambda r: r['filed'])['accn']
    base = f"https://www.sec.gov/Archives/edgar/data/{int(companyfacts['cik'])}/{accn.replace('-', '')}/"
    names = [i['name'] for i in session.get(base + 'index.json', timeout=30).json()['directory']['item']]
    # most filings ship a separate _cal.xml; some (Microsoft) embed the linkbases in the .xsd schema
    candidates = [n for n in names if n.endswith('_cal.xml')] or [n for n in names if n.endswith('.xsd')]
    for name in candidates:
        text = session.get(base + name, timeout=60).text
        time.sleep(0.15)
        if 'calculationLink' in text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
            return text
    return None


def _concept(href: str) -> str:
    """'...#us-gaap_Revenues' -> 'Revenues'; a company's own concept keeps its prefix: 'bac:FinancingReceivable...'."""
    prefix, _, local = href.split('#')[-1].partition('_')
    return local if prefix == 'us-gaap' else f'{prefix}:{local}'


def parse_calc(xml: str) -> dict[str, dict[str, list[tuple[float, str]]]]:
    """role -> parent -> [(weight, child)]. A role can be split over several calculationLink blocks; they are merged."""
    roles: dict[str, dict[str, list[tuple[float, str]]]] = {}
    for link in BeautifulSoup(xml, 'xml').find_all('calculationLink'):
        tree = roles.setdefault(link.get('xlink:role', '').split('/')[-1], {})
        loc = {l['xlink:label']: _concept(l['xlink:href']) for l in link.find_all('loc')}
        for arc in link.find_all('calculationArc'):
            tree.setdefault(loc[arc['xlink:from']], []).append((float(arc['weight']), loc[arc['xlink:to']]))
    return roles


def income_statement(roles: dict) -> dict[str, list[tuple[float, str]]] | None:
    """The income statement's tree: named like one, not comprehensive income or a detail note, with net income on top."""
    best = None
    for role, tree in roles.items():
        name = role.upper()
        if any(w in name for w in ('COMPREHENSIVE', 'PARENTHETICAL', 'DETAIL', 'TABLE', 'POLICIES')):
            continue
        if not any(w in name for w in ('INCOME', 'OPERATIONS', 'EARNINGS')) or not any(t in tree for t in TOPS):
            continue
        if best is None or len(tree) > len(best):
            best = tree
    return best


def _children(tree: dict, node: str, sign: int) -> list[str]:
    return [c for w, c in tree.get(node, []) if (w > 0) == (sign > 0)]


def _one(items: list[str]) -> str | None:
    return items[0] if len(items) == 1 else None


def concepts_on(tree: dict) -> set[str]:
    return set(tree) | {c for kids in tree.values() for _, c in kids}


def positions(tree: dict) -> dict[str, str]:
    """
    Where each metric sits in the company's own arithmetic, whatever its concept is called:
    revenue is the + line under gross profit, else under operating income, else (a bank) under pre-tax income;
    pre-tax income is the line named ...BeforeIncomeTaxes... that the net income chain adds;
    a bank's provision is the - line under pre-tax income (or under net interest income after provision) that is not
    non-interest expense. A position with more than one candidate is left out rather than guessed.
    """
    nodes = concepts_on(tree)
    out = {}
    pretax = [n for n in nodes if 'BeforeIncomeTaxes' in n and 'Domestic' not in n and 'Foreign' not in n]
    if len(pretax) == 1:
        out['pretax_income'] = pretax[0]

    if 'GrossProfit' in tree:
        revenue = _one(_children(tree, 'GrossProfit', +1))
        cost = _one(_children(tree, 'GrossProfit', -1))
        if cost:
            out['cost_of_revenue'] = cost
    elif 'OperatingIncomeLoss' in tree:
        revenue = _one([c for c in _children(tree, 'OperatingIncomeLoss', +1) if c not in SUBTOTALS])
    elif 'pretax_income' in out:
        # a bank: pre-tax income = + revenue - provision - non-interest expense
        revenue = _one([c for c in _children(tree, out['pretax_income'], +1) if c in tree])
        provision = [c for c in _children(tree, out['pretax_income'], -1) if c != 'NoninterestExpense']
        provision += _children(tree, 'InterestIncomeExpenseAfterProvisionForLoanLoss', -1)
        if _one(provision):
            out['provision_for_credit_losses'] = provision[0]
    else:
        revenue = None
    if revenue:
        out['revenue'] = revenue
    return out


def cost_group(tree: dict, concept: str) -> list[str] | None:
    """
    The lines a cost of revenue competes with: its siblings under a 'costs and expenses' group, or the other - lines of
    operating income. None if the concept is the - line under gross profit, which needs no comparison.
    """
    if concept in _children(tree, 'GrossProfit', -1):
        return None
    for parent, kids in tree.items():
        names = [c for _, c in kids]
        if concept in names and parent not in SUBTOTALS:
            return names
    if concept in _children(tree, 'OperatingIncomeLoss', -1):
        return _children(tree, 'OperatingIncomeLoss', -1)
    return []
