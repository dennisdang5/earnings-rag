import json
import sys
import time
from datetime import date, timedelta

from earnings_rag.config import settings

MIN_FISCAL_YEAR = 2020

# metric -> the XBRL concept each company uses for it. Hand-verified against the real data: the "standard" concept
# is not always right (Capital One's RevenueFromContractWithCustomer... is only its fee revenue, ~1/7 of Revenues).
# additive=False for per-share values: share counts change, so FY EPS minus 9-month EPS is not Q4 EPS.
METRICS = {
    'revenue': {
        'concepts': {'NVDA': 'Revenues', 'COF': 'Revenues',
                     'AAPL': 'RevenueFromContractWithCustomerExcludingAssessedTax'},
        'unit': 'USD', 'additive': True},
    'net_income': {
        'concepts': {t: 'NetIncomeLoss' for t in ('NVDA', 'AAPL', 'COF')},
        'unit': 'USD', 'additive': True},
    'eps_diluted': {
        'concepts': {t: 'EarningsPerShareDiluted' for t in ('NVDA', 'AAPL', 'COF')},
        'unit': 'USD/shares', 'additive': False},
    'operating_cash_flow': {
        'concepts': {t: 'NetCashProvidedByUsedInOperatingActivities' for t in ('NVDA', 'AAPL', 'COF')},
        'unit': 'USD', 'additive': True},
    'gross_profit': {
        'concepts': {t: 'GrossProfit' for t in ('NVDA', 'AAPL')},
        'unit': 'USD', 'additive': True},
    'operating_income': {
        'concepts': {t: 'OperatingIncomeLoss' for t in ('NVDA', 'AAPL')},
        'unit': 'USD', 'additive': True},
    'rnd_expense': {
        'concepts': {t: 'ResearchAndDevelopmentExpense' for t in ('NVDA', 'AAPL')},
        'unit': 'USD', 'additive': True},
    'net_interest_income': {
        'concepts': {'COF': 'InterestIncomeExpenseNet'},
        'unit': 'USD', 'additive': True},
    'noninterest_income': {
        'concepts': {'COF': 'NoninterestIncome'},
        'unit': 'USD', 'additive': True},
}

PERIOD_ORDER = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4, 'FY': 5}

# Companies report year-to-date totals, never Q4 on its own: Q4 = FY - 9 months, and so on.
# period -> (year-to-date total, the earlier total to subtract)
DERIVATIONS = {'Q2': ('H1', 'Q1'), 'Q3': ('M9', 'H1'), 'Q4': ('FY', 'M9')}


def fetch_companyfacts(ticker: str, refresh: bool = False) -> dict:
    """Download (and cache, like the 10-K files) the SEC's XBRL facts for one company."""
    path = settings.data_dir / 'xbrl' / f'{ticker}.json'
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding='utf-8'))

    # Imported here because ingest raises at import time when SEC_USER_AGENT is unset (as in CI),
    # and normalize() must stay importable without it.
    from earnings_rag.ingest import session, load_ticker_map

    cik = load_ticker_map()[ticker]
    r = session.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json', timeout=60)
    r.raise_for_status()
    data = r.json()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    time.sleep(0.15)
    return data


def _days(row: dict) -> int:
    return (date.fromisoformat(row['end']) - date.fromisoformat(row['start'])).days


def _classify(days: int) -> str | None:
    """Duration shape: 3 months, 6 / 9 months year-to-date, or a full year."""
    if 80 <= days <= 100:
        return 'Q'
    if 170 <= days <= 195:
        return 'H1'
    if 260 <= days <= 285:
        return 'M9'
    if 350 <= days <= 380:
        return 'FY'
    return None


def _latest_unique(rows: list[dict]) -> list[dict]:
    """
    Each value is repeated in later filings as a prior-period comparison. Keep one row per period: the most
    recently filed, so restatements win. Instant facts have no 'start' and are skipped.

    Caution for per-share values: a period is only re-presented for about two years, so older periods keep the
    share basis of their original filing and are NOT reliably split-adjusted (see DECISIONS.md).
    """
    best = {}
    for r in rows:
        if 'start' not in r:
            continue
        key = (r['start'], r['end'])
        if key not in best or r['filed'] >= best[key]['filed']:
            best[key] = r
    return list(best.values())


def _locate(start: date, end: date, years: list[tuple]) -> tuple[int, date] | None:
    """
    Which fiscal year does a period fall in? Uses the fact's own dates, NOT the fy/fp fields, which describe the
    filing that reported the fact (a prior-year comparison carries next year's fy). Periods after the last
    annual fact belong to the fiscal year still in progress.
    """
    for fy, fy_start, fy_end in years:
        if fy_start <= start and end <= fy_end:
            return fy, fy_start
    fy, _, last_end = years[-1]
    if start > last_end:
        return fy + 1, last_end + timedelta(days=1)
    return None


def _normalize_metric(rows: list[dict], ticker: str, metric: str, concept: str, spec: dict) -> list[dict]:
    rows = _latest_unique(rows)
    annual = {(r['start'], r['end']) for r in rows if _classify(_days(r)) == 'FY'}
    if not annual:
        return []
    # fiscal year = the calendar year it ends in (NVIDIA's fiscal 2025 ended 2025-01-26)
    years = sorted((date.fromisoformat(e).year, date.fromisoformat(s), date.fromisoformat(e)) for s, e in annual)

    by_year: dict[int, dict[str, dict]] = {}
    for r in rows:
        kind = _classify(_days(r))
        if kind is None:
            continue
        start, end = date.fromisoformat(r['start']), date.fromisoformat(r['end'])
        located = _locate(start, end, years)
        if located is None or located[0] < MIN_FISCAL_YEAR:
            continue
        fy, fy_start = located
        offset = (start - fy_start).days

        if kind == 'Q':
            label = f'Q{round(offset / 91.3) + 1}'  # quarters start ~0, 91, 182, 273 days into the year
            if label not in PERIOD_ORDER:
                continue
        elif offset > 10:
            continue  # year-to-date and annual periods must begin at the fiscal year start
        else:
            label = kind
        by_year.setdefault(fy, {})[label] = r

    def emit(fy, period, src, value, start, end, derived):
        return {'ticker': ticker, 'metric': metric, 'segment': '', 'concept': concept,
                'fiscal_year': fy, 'fiscal_period': period, 'period_start': start, 'period_end': end,
                'value': value, 'unit': spec['unit'], 'derived': derived,
                'form': src['form'], 'accession': src['accn'], 'filed': src['filed']}

    out = []
    for fy, cum in by_year.items():
        for period in PERIOD_ORDER:
            if period in cum:
                r = cum[period]
                out.append(emit(fy, period, r, r['val'], r['start'], r['end'], False))

        if not spec['additive']:
            continue
        for period, (total, earlier) in DERIVATIONS.items():
            if period in cum or total not in cum or earlier not in cum:
                continue
            t, e = cum[total], cum[earlier]
            start = (date.fromisoformat(e['end']) + timedelta(days=1)).isoformat()
            out.append(emit(fy, period, t, t['val'] - e['val'], start, t['end'], True))
    return out


def normalize(companyfacts: dict, ticker: str) -> list[dict]:
    """
    Turn the SEC's raw facts into one clean row per (metric, fiscal year, fiscal period), with missing quarters
    derived by subtraction and flagged derived=True. Pure function: no network, no database.
    """
    gaap = companyfacts['facts']['us-gaap']
    out = []
    for metric, spec in METRICS.items():
        concept = spec['concepts'].get(ticker)
        if concept is None or concept not in gaap:
            continue
        out.extend(_normalize_metric(gaap[concept]['units'].get(spec['unit'], []), ticker, metric, concept, spec))
    out.sort(key=lambda r: (r['metric'], r['fiscal_year'], PERIOD_ORDER[r['fiscal_period']]))
    return out


def ingest_facts(refresh: bool = False) -> None:
    from earnings_rag.store import init_schema, upsert_facts

    init_schema()
    for ticker in settings.tickers:
        rows = normalize(fetch_companyfacts(ticker, refresh), ticker)
        upsert_facts(rows)
        print(f'{ticker}: {len(rows)} facts ({sum(r["derived"] for r in rows)} derived)')


if __name__ == '__main__':
    ingest_facts(refresh='--refresh' in sys.argv)
