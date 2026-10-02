import re
from datetime import date

from bs4 import BeautifulSoup

from earnings_rag.config import settings
from earnings_rag.xbrl import METRICS, _classify

# The three standard axes that carry breakdowns, the same for every company.
AXES = {'ProductOrServiceAxis': 'product',
        'StatementBusinessSegmentsAxis': 'segment',
        'StatementGeographicalAxis': 'geography'}

CORPORATE = 'Corporate and other'  # reconciling item: segments only add up to the total with it
SMALL_WORDS = {'And', 'Of', 'The'}
SUM_TOLERANCE_M = 1       # millions
MAX_SUBSET_MEMBERS = 20   # the subset-sum check is skipped for axes with more members than this


def member_name(member: str) -> str:
    """
    'nvda:ComputeAndNetworkingSegmentMember' -> 'Compute and Networking'. Companies rename members between years
    (ComputeAndNetworkingMember in the FY2023 10-K, ...SegmentMember in FY2025), so the suffixes are stripped and
    both spellings become one series. A heuristic, not the filing's own label: 'IPhoneMember' -> 'I Phone'.
    """
    name = member.split(':')[-1]
    if name == 'CorporateNonSegmentMember':
        return CORPORATE
    name = re.sub(r'Member$', '', name)
    name = re.sub(r'Segment$', '', name)
    if re.fullmatch(r'[A-Z]{2}', name):  # ISO country codes (US, CN, TW)
        return name
    words = re.split(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', name)
    return ' '.join(w.lower() if w in SMALL_WORDS and i else w for i, w in enumerate(words))


def _axis_and_member(dims: list[tuple[str, str]]) -> tuple[str, str] | None:
    """
    ('', '') for an undimensioned (consolidated) fact, (axis, member name) for a one-axis breakdown, None for anything
    else. Since ASU 2023-07 business segments carry a second dimension, ConsolidationItemsAxis=OperatingSegmentsMember
    (Apple's FY2022 filing has the segment alone, its FY2025 filing has both), so that member is a qualifier and is
    dropped. If nothing else is left it is the total of the operating segments ('total', ''), not a consolidated
    figure; parse_inline uses it to catch a corporate item that merely repeats it.
    """
    rest, qualified = [], False
    for axis, member in dims:
        if axis == 'ConsolidationItemsAxis':
            if member.endswith('OperatingSegmentsMember'):
                qualified = True
                continue
            if member.endswith('CorporateNonSegmentMember'):
                rest.append(('StatementBusinessSegmentsAxis', member))
                continue
            return None  # eliminations and other reconciling items: not a breakdown we model
        rest.append((axis, member))

    if not dims:
        return '', ''
    if qualified and not rest:
        return 'total', ''
    if len(rest) != 1 or rest[0][0] not in AXES:
        return None  # several remaining dimensions (fees by product by segment) or an axis we don't model
    return AXES[rest[0][0]], member_name(rest[0][1])


def _number(tag) -> float | None:
    """The value as the filing displays it, with its inline scale and sign applied. None if not numeric."""
    if tag.get('xsi:nil') == 'true' or tag.get('nil') == 'true':
        return None
    text = tag.get_text(strip=True).replace(',', '')
    if 'fixed-zero' in (tag.get('format') or '') or text in ('—', '-', '–'):
        value = 0.0
    else:
        try:
            value = float(text)
        except ValueError:
            return None  # e.g. ixt-sec:numwordsen ("three")
    value *= 10 ** int(tag.get('scale') or 0)
    return -value if tag.get('sign') == '-' else value


def parse_inline(html: bytes, ticker: str) -> list[dict]:
    """
    Every full-year fact in a 10-K's inline XBRL for a metric in METRICS, as one row per (metric, axis, member, year).
    axis '' means consolidated. Pure: no network, no database.
    """
    soup = BeautifulSoup(html, 'xml')

    contexts = {}
    for c in soup.find_all('context'):
        period = c.find('period')
        start, end = period.find('startDate'), period.find('endDate')
        if start is None or end is None:
            continue  # instant (balance sheet) context
        dims = [(m['dimension'].split(':')[-1], m.get_text(strip=True)) for m in c.find_all('explicitMember')]
        if c.find('typedMember') is not None:
            continue
        contexts[c['id']] = (dims, start.get_text(strip=True), end.get_text(strip=True))

    concept_metric = {spec['concepts'][ticker]: metric for metric, spec in METRICS.items()
                      if ticker in spec['concepts'] and spec['unit'] == 'USD'}
    found = {}
    for tag in soup.find_all('nonFraction'):
        concept = tag.get('name', '').split(':')[-1]
        if concept not in concept_metric or tag.get('contextRef') not in contexts:
            continue
        dims, start, end = contexts[tag['contextRef']]
        if _classify((date.fromisoformat(end) - date.fromisoformat(start)).days) != 'FY':
            continue
        located = _axis_and_member(dims)
        value = _number(tag)
        if located is None or value is None:
            continue
        axis, member = located
        # the same fact is often printed twice (statement and note); the first reading is kept
        found.setdefault((concept_metric[concept], axis, member, start, end),
                         {'metric': concept_metric[concept], 'concept': concept, 'axis': axis, 'segment': member,
                          'value': value, 'period_start': start, 'period_end': end})

    # NVIDIA's FY2026 10-K tags a second "corporate" operating income equal to the total of the operating segments
    # (139,297 where the earlier filings give -6,507). A reconciling item that repeats the total is a tagging artifact.
    totals = {(f['metric'], f['period_start'], f['period_end']): f['value'] for f in found.values()
              if f['axis'] == 'total'}
    return [f for f in found.values() if f['axis'] != 'total'
            and not (f['segment'] == CORPORATE and f['value'] != 0
                     and totals.get((f['metric'], f['period_start'], f['period_end'])) == f['value'])]


def _filing_for(companyfacts: dict, ticker: str, report_date: str) -> dict | None:
    """The 10-K (accession, filed) that first reported this fiscal year end. The HTML does not say."""
    gaap = companyfacts['facts']['us-gaap']
    best = None
    for spec in METRICS.values():
        for r in gaap.get(spec['concepts'].get(ticker), {}).get('units', {}).get(spec['unit'], []):
            if r['end'] == report_date and r['form'] == '10-K' and (best is None or r['filed'] < best['filed']):
                best = r
    return best


def normalize_segments(parsed: dict[str, list[dict]], companyfacts: dict, ticker: str) -> list[dict]:
    """
    parsed: {report date: parse_inline rows} per 10-K. One facts row per (metric, axis, segment, fiscal year) for the
    breakdown rows; the latest filing wins, as in xbrl.normalize, so a reorganized prior year takes the newest figures.
    """
    out = {}
    for report_date in sorted(parsed):
        filing = _filing_for(companyfacts, ticker, report_date)
        if filing is None:
            continue
        for f in parsed[report_date]:
            if not f['axis']:
                continue
            unit = METRICS[f['metric']]['unit']
            out[(f['metric'], f['axis'], f['segment'], f['period_end'][:4])] = {
                **f, 'ticker': ticker, 'fiscal_year': int(f['period_end'][:4]), 'fiscal_period': 'FY',
                'unit': unit, 'derived': False, 'form': filing['form'], 'accession': filing['accn'],
                'filed': filing['filed'], 'split_factor': 1}
    return sorted(out.values(), key=lambda r: (r['metric'], r['axis'], r['fiscal_year'], r['segment']))


def cross_check(parsed: dict[str, list[dict]], companyfacts: dict, ticker: str) -> tuple[int, list[dict]]:
    """
    Tests the parser (scale, sign, number formats) against an independent source: every consolidated value read from
    the HTML must equal the SEC API's value for the same concept and period. Returns (checked, mismatches).
    """
    gaap = companyfacts['facts']['us-gaap']
    checked, bad = 0, []
    for facts in parsed.values():
        for f in facts:
            if f['axis']:
                continue
            readings = {r['val'] for r in gaap.get(f['concept'], {}).get('units', {}).get('USD', [])
                        if r.get('start') == f['period_start'] and r['end'] == f['period_end']}
            if not readings:
                continue
            checked += 1
            if not any(abs(f['value'] - v) <= 1 for v in readings):
                bad.append({'metric': f['metric'], 'period_end': f['period_end'], 'parsed': f['value'],
                            'api': sorted(readings)})
    return checked, bad


def sum_check(rows: list[dict], consolidated: dict[tuple, float]) -> tuple[int, list[tuple]]:
    """
    Does each breakdown reconcile to the consolidated total? Axes are hierarchical (Data Center = Compute + Networking),
    so the test is whether SOME subset of the members sums to the total, within a million. consolidated maps
    (metric, fiscal_year) -> value. Returns (axes checked, the (metric, axis, year) that did not reconcile).
    """
    groups: dict[tuple, list[int]] = {}
    for r in rows:
        groups.setdefault((r['metric'], r['axis'], r['fiscal_year']), []).append(round(r['value'] / 1e6))

    checked, failed = 0, []
    for (metric, axis, fy), values in groups.items():
        total = consolidated.get((metric, fy))
        if total is None or len(values) > MAX_SUBSET_MEMBERS:
            continue
        checked += 1
        sums = {0}
        for v in values:
            sums |= {s + v for s in sums}
        if not any(abs(s - round(total / 1e6)) <= SUM_TOLERANCE_M for s in sums if s):
            failed.append((metric, axis, fy))
    return checked, failed


def ingest_segments(ticker: str, companyfacts: dict) -> bool:
    """Parse the cached 10-Ks, validate, store. Returns False (storing nothing) if the parser disagrees with the API."""
    from earnings_rag import xbrl
    from earnings_rag.store import upsert_facts

    parsed = {p.stem: parse_inline(p.read_bytes(), ticker) for p in sorted((settings.raw_dir / ticker).glob('*.html'))}
    checked, bad = cross_check(parsed, companyfacts, ticker)
    print(f'{ticker}: segment parser check: {checked} consolidated values vs the SEC API, {len(bad)} mismatches')
    for b in bad[:5]:
        print(f'  mismatch {b}')
    if bad:
        print(f'{ticker}: segment parser disagrees with the API, segments not stored')
        return False

    rows = normalize_segments(parsed, companyfacts, ticker)
    consolidated = {(r['metric'], r['fiscal_year']): r['value'] for r in xbrl.normalize(companyfacts, ticker)
                    if r['fiscal_period'] == 'FY'}
    axes_checked, failed = sum_check(rows, consolidated)
    print(f'{ticker}: {len(rows)} segment facts, {axes_checked - len(failed)}/{axes_checked} breakdowns reconcile '
          f'to the total')
    for metric, axis, fy in failed:
        print(f'  does not reconcile: {metric} by {axis} FY{fy}')
    upsert_facts(rows)
    return True
