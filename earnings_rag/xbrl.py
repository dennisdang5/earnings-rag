import json
import sys
import time
from datetime import date, timedelta
from statistics import median

from earnings_rag.config import settings

MIN_FISCAL_YEAR = 2020

# metric -> the XBRL concept each company uses for it. Hand-verified against the real data: the "standard" concept
# is not always right (Capital One's RevenueFromContractWithCustomer... is only its fee revenue, ~1/7 of Revenues).
# additive=False for per-share values: share counts change, so FY EPS minus 9-month EPS is not Q4 EPS.
# per_share=True values are rescaled to today's share count (see find_splits).
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
        'unit': 'USD/shares', 'additive': False, 'per_share': True},
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

SHARES_CONCEPT = 'WeightedAverageNumberOfDilutedSharesOutstanding'

# A jump in the cover-page share count within SPLIT_TOLERANCE of one of these ratios (or its reciprocal, for reverse
# splits) is a split candidate; ordinary buybacks and issuance move the count a few percent per quarter.
SPLIT_RATIOS = [1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50]
SPLIT_TOLERANCE = 0.06
CORROBORATION_TOLERANCE = 0.02

# ticker -> [(cutoff, ratio)] used instead of detection, for issuers whose cover-page counts are not in the data
# (multi-class companies such as GOOGL). A cutoff is the filing date of the first post-split filing.
SPLIT_OVERRIDES: dict[str, list[tuple[date, float]]] = {}

EPS_ROUNDING = 0.005       # EPS is reported to 2 decimals; scaling a reading by 1/factor scales this error too
MIN_SPLIT_SIZED = 1.8      # readings disagreeing by at least this, in a split-like ratio, point to a wrong cutoff
MAX_MISMATCH_RATE = 0.05

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


def _latest_unique(rows: list[dict], splits: list[tuple[date, float]] | None = None) -> list[dict]:
    """
    Each value is repeated in later filings as a prior-period comparison. Keep one row per period: the most
    recently filed, so restatements win. Instant facts have no 'start' and are skipped.

    Per-share values are the exception to "latest wins": a period is only re-presented for about two years, so
    older ones keep their original share basis. They are rescaled in emit() using find_splits().

    Given splits, a per-share period takes the most precise reported reading instead (see _most_precise).
    """
    readings: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if 'start' in r:
            readings.setdefault((r['start'], r['end']), []).append(r)
    out = []
    for group in readings.values():
        latest = max(group, key=lambda r: r['filed'])
        out.append(_most_precise(group, latest, splits) if splits is not None else latest)
    return out


def _most_precise(group: list[dict], latest: dict, splits: list[tuple[date, float]]) -> dict:
    """
    EPS is published to 2 decimals, so each split divides the precision away: NVIDIA's FY2023 EPS was 1.74 before the
    10:1 split and 0.17 after, but 1.74 / 10 = 0.174 is the company's own figure with a digit more. Take the reading
    filed before the most splits, as long as it agrees with the latest one within the latest's rounding; if it does
    not (a restatement), the latest reading wins. Nothing is computed: every value is one the company reported.
    """
    def factor(r):
        return split_factor(r['filed'], splits)

    oldest = max(group, key=factor)
    if factor(oldest) <= factor(latest):
        return latest
    if abs(oldest['val'] / factor(oldest) - latest['val'] / factor(latest)) <= EPS_ROUNDING / factor(latest) + 1e-9:
        return oldest
    return latest


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


def _cover_counts(companyfacts: dict) -> list[tuple[str, float]]:
    """(filing date, shares outstanding on the cover page) per filing, oldest first. Share classes are summed."""
    rows = (companyfacts['facts'].get('dei', {}).get('EntityCommonStockSharesOutstanding', {})
            .get('units', {}).get('shares', []))
    per_filing: dict[tuple[str, str], float] = {}
    for r in rows:
        key = (r['filed'], r['accn'])
        per_filing[key] = per_filing.get(key, 0) + r['val']
    return sorted((filed, v) for (filed, _), v in per_filing.items() if v > 0)  # zeros are placeholder rows


def _nearest_split_ratio(jump: float) -> float | None:
    """The split ratio a share-count jump looks like (0.1 for a 1-for-10 reverse split), or None for ordinary drift."""
    for ratio in SPLIT_RATIOS:
        if abs(jump / ratio - 1) < SPLIT_TOLERANCE:
            return ratio
        if abs(jump * ratio - 1) < SPLIT_TOLERANCE:
            return 1 / ratio
    return None


def _corroborated(cutoff: date, ratio: float, companyfacts: dict) -> bool:
    """
    A real split restates old periods: a period reported both before and after the cutoff shows diluted shares scaled
    by the ratio. An acquisition that doubles the share count restates nothing, so the same period reads ~1x.
    With no period straddling the cutoff yet, nothing contradicts the jump, so it is accepted.
    """
    rows = companyfacts['facts']['us-gaap'].get(SHARES_CONCEPT, {}).get('units', {}).get('shares', [])
    by_period: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        if 'start' in r and r['val'] > 0:
            by_period.setdefault((r['start'], r['end']), {})[r['filed']] = r['val']

    observed = []
    for readings in by_period.values():
        before = [f for f in readings if date.fromisoformat(f) < cutoff]
        after = [f for f in readings if date.fromisoformat(f) >= cutoff]
        if before and after:  # the closest reading on each side, so only this split lies between them
            observed.append(readings[min(after)] / readings[max(before)])

    return not observed or abs(median(observed) / ratio - 1) < CORROBORATION_TOLERANCE


def find_splits(companyfacts: dict, ticker: str | None = None) -> list[tuple[date, float]]:
    """
    Stock splits as (cutoff, ratio), oldest first; a value filed before the cutoff is on the old share basis.

    Detected from the share count on each filing's cover page, which jumps by the split ratio between the last
    pre-split and first post-split filing, so the cutoff is exactly the first filing after the split. Tagged split
    ratios are not used: some companies tag them in the first post-split filing, some months late (WMT), some before
    the split happens (GOOGL).
    """
    if ticker in SPLIT_OVERRIDES:
        return sorted(SPLIT_OVERRIDES[ticker])

    counts = _cover_counts(companyfacts)
    splits = []
    for (_, before), (filed, after) in zip(counts, counts[1:]):
        ratio = _nearest_split_ratio(after / before)
        cutoff = date.fromisoformat(filed)
        if ratio is not None and _corroborated(cutoff, ratio, companyfacts):
            splits.append((cutoff, ratio))
    return splits


def split_factor(filed: str, splits: list[tuple[date, float]]) -> float:
    """Product of the ratios of every split after this value's filing date: its per-share value is that much too high."""
    factor = 1
    for cutoff, ratio in splits:
        if date.fromisoformat(filed) < cutoff:  # a filing on the cutoff date reports the ratio, so is post-split
            factor *= ratio
    return factor


def _normalize_metric(rows: list[dict], ticker: str, metric: str, concept: str, spec: dict,
                      splits: list[tuple[date, float]]) -> list[dict]:
    rows = _latest_unique(rows, splits if spec.get('per_share') else None)
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
        factor = split_factor(src['filed'], splits) if spec.get('per_share') else 1
        if factor != 1:
            value = round(value / factor, 6)
        return {'ticker': ticker, 'metric': metric, 'segment': '', 'axis': '', 'concept': concept,
                'fiscal_year': fy, 'fiscal_period': period, 'period_start': start, 'period_end': end,
                'value': value, 'unit': spec['unit'], 'derived': derived,
                'form': src['form'], 'accession': src['accn'], 'filed': src['filed'], 'split_factor': factor}

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
    splits = find_splits(companyfacts, ticker)
    out = []
    for metric, spec in METRICS.items():
        concept = spec['concepts'].get(ticker)
        if concept is None or concept not in gaap:
            continue
        out.extend(_normalize_metric(gaap[concept]['units'].get(spec['unit'], []), ticker, metric, concept, spec,
                                       splits))
    out.sort(key=lambda r: (r['metric'], r['fiscal_year'], PERIOD_ORDER[r['fiscal_period']]))
    return out


def check_split_consistency(companyfacts: dict, ticker: str, splits: list[tuple[date, float]]) -> tuple[int, list[dict]]:
    """
    A period reported by several filings must agree once every reading is on the same share basis. Returns the
    number of per-share periods checked and the ones that disagree by more than the readings' rounding.

    Each mismatch is classified: 'split' if the readings differ by about a split ratio (what a wrong or missing
    cutoff produces, e.g. 20x), otherwise 'restated' (a genuine restatement, e.g. Tesla's Q1 2024 EPS 0.34 -> 0.41).
    Only the former should block ingestion.
    """
    gaap = companyfacts['facts']['us-gaap']
    checked, bad = 0, []
    for metric, spec in METRICS.items():
        concept = spec['concepts'].get(ticker)
        if not spec.get('per_share') or concept not in gaap:
            continue
        by_period: dict[tuple[str, str], dict[str, float]] = {}
        for r in gaap[concept]['units'].get(spec['unit'], []):
            if 'start' in r and r['end'] >= f'{MIN_FISCAL_YEAR - 1}-01-01':
                by_period.setdefault((r['start'], r['end']), {})[r['filed']] = r['val']

        for period, readings in by_period.items():
            if len(readings) < 2:
                continue
            checked += 1
            factors = {f: split_factor(f, splits) for f in readings}
            adjusted = [v / factors[f] for f, v in readings.items()]
            errors = sorted((EPS_ROUNDING / factor for factor in factors.values()), reverse=True)
            if max(adjusted) - min(adjusted) <= errors[0] + errors[1] + 1e-9:  # the two largest rounding errors
                continue

            low, high = min(abs(v) for v in adjusted), max(abs(v) for v in adjusted)
            ratio = high / low if low else float('inf')
            split_sized = ratio >= MIN_SPLIT_SIZED and _nearest_split_ratio(ratio) is not None
            bad.append({'metric': metric, 'period': period, 'readings': readings,
                        'kind': 'split' if split_sized else 'restated'})
    return checked, bad


def ingest_facts(refresh: bool = False) -> None:
    from earnings_rag.store import init_schema, upsert_facts

    init_schema()
    failed = []
    for ticker in settings.tickers:
        facts = fetch_companyfacts(ticker, refresh)

        checked, bad = check_split_consistency(facts, ticker, find_splits(facts, ticker))
        split_sized = [b for b in bad if b['kind'] == 'split']
        print(f'{ticker}: split check: {checked} periods, {len(split_sized)} split-sized mismatches, '
              f'{len(bad) - len(split_sized)} restatements')
        for b in split_sized[:5]:
            print(f'  mismatch {b["metric"]} {b["period"]}: {b["readings"]}')
        if checked and len(split_sized) / checked > MAX_MISMATCH_RATE:
            print(f'{ticker}: split adjustment looks wrong, not stored')
            failed.append(ticker)
            continue

        rows = normalize(facts, ticker)
        upsert_facts(rows)
        print(f'{ticker}: {len(rows)} facts ({sum(r["derived"] for r in rows)} derived)')

        from earnings_rag.segments import ingest_segments  # segments imports this module, so not at the top
        if not ingest_segments(ticker, facts):
            failed.append(ticker)

    if failed:
        raise SystemExit(f'ingestion check failed for: {", ".join(failed)}')


if __name__ == '__main__':
    ingest_facts(refresh='--refresh' in sys.argv)
