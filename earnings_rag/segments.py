import re
from datetime import date

from bs4 import BeautifulSoup

from earnings_rag.config import settings
from earnings_rag.resolver import resolve
from earnings_rag.xbrl import METRICS, _classify, fiscal_calendar, period_label

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


def parse_filing(html: bytes, concepts: dict[str, str],
                 durations: tuple[str, ...] = ('FY',)) -> tuple[list[dict], list[dict]]:
    """
    Every full-year fact (and fiscal-year-end balance) in a 10-K's inline XBRL for a metric in METRICS, one row per (metric, axis, member, year).
    axis '' means consolidated. Pure: no network, no database.

    durations: the period lengths to keep, as xbrl._classify names them. A 10-Q passes ('Q',): its three-month figures
    are read directly, and the six- and nine-month year-to-date ones are dropped (a quarter is never derived from them).
    A 10-Q has no full-year duration, so no fiscal year end is found and balances are not read from it.

    Also returns the dollar items tagged on the corporate member alone (ConsolidationItemsAxis=CorporateNonSegmentMember,
    nothing else): stock compensation, unallocated expenses, acquisition costs, but also interest and other non-operating
    income. add_corporate_remainder matches them against a missing corporate figure.
    """
    soup = BeautifulSoup(html, 'xml')

    contexts = {}
    for c in soup.find_all('context'):
        if c.find('typedMember') is not None:
            continue
        period = c.find('period')
        start, end, instant = period.find('startDate'), period.find('endDate'), period.find('instant')
        dims = [(m['dimension'].split(':')[-1], m.get_text(strip=True)) for m in c.find_all('explicitMember')]
        if instant is not None:
            contexts[c['id']] = (dims, None, instant.get_text(strip=True))
        elif start is not None and end is not None:
            contexts[c['id']] = (dims, start.get_text(strip=True), end.get_text(strip=True))

    # A balance (loans, deposits) is read only at a fiscal year end, which a full-year duration in the same filing
    # identifies; other dates in the notes (a debt maturity, a mid-year snapshot) are not fiscal-year figures.
    year_ends = {end for _, start, end in contexts.values()
                 if start is not None and _classify((date.fromisoformat(end) - date.fromisoformat(start)).days) == 'FY'}

    # one concept per metric, the one the company uses (resolver.primary): reading a smaller look-alike concept as well
    # would collide with it on the same slice
    concept_metric = {concept: metric for metric, concept in concepts.items() if METRICS[metric]['unit'] == 'USD'}
    found = {}
    for tag in soup.find_all('nonFraction'):
        concept = tag.get('name', '').split(':')[-1]
        if concept not in concept_metric or tag.get('contextRef') not in contexts:
            continue
        metric = concept_metric[concept]
        dims, start, end = contexts[tag['contextRef']]
        if METRICS[metric].get('balance'):
            if start is not None or end not in year_ends:
                continue
            start = end  # a balance has no start; stored as a one-day period like the consolidated balance rows
        elif start is None or _classify((date.fromisoformat(end) - date.fromisoformat(start)).days) not in durations:
            continue
        located = _axis_and_member(dims)
        value = _number(tag)
        if located is None or value is None:
            continue
        axis, member = located
        raw = next((m for a, m in dims if a in AXES), '')  # 'aapl:IPhoneMember', for its label in the linkbase
        # the same fact is often printed twice (statement and note); the first reading is kept
        found.setdefault((metric, axis, member, start, end),
                         {'metric': metric, 'concept': concept, 'axis': axis, 'segment': member, 'member': raw,
                          'value': value, 'period_start': start, 'period_end': end})

    # NVIDIA's FY2026 10-K tags a second "corporate" operating income equal to the total of the operating segments
    # (139,297 where the earlier filings give -6,507). A reconciling item that repeats the total is a tagging artifact.
    totals = {(f['metric'], f['period_start'], f['period_end']): f['value'] for f in found.values()
              if f['axis'] == 'total'}
    rows = [f for f in found.values() if f['axis'] != 'total'
            and not (f['segment'] == CORPORATE and f['value'] != 0
                     and totals.get((f['metric'], f['period_start'], f['period_end'])) == f['value'])]

    items = {}
    for tag in soup.find_all('nonFraction'):
        unit = (tag.get('unitRef') or '').lower()
        if tag.get('contextRef') not in contexts or 'usd' not in unit or 'share' in unit:
            continue
        dims, start, end = contexts[tag['contextRef']]
        if (start is None or len(dims) != 1 or dims[0][0] != 'ConsolidationItemsAxis'
                or not dims[0][1].endswith('CorporateNonSegmentMember')
                or _classify((date.fromisoformat(end) - date.fromisoformat(start)).days) not in durations):
            continue
        value = _number(tag)
        concept = tag.get('name', '').split(':')[-1]
        if value is not None:
            items.setdefault((concept, start, end),
                             {'concept': concept, 'value': value, 'period_start': start, 'period_end': end})
    return rows, list(items.values())


def parse_inline(html: bytes, concepts: dict[str, str], durations: tuple[str, ...] = ('FY',)) -> list[dict]:
    """parse_filing's facts rows without the corporate items."""
    return parse_filing(html, concepts, durations)[0]


def _filing_for(companyfacts: dict, concepts: dict[str, str], report_date: str, form: str = '10-K') -> dict | None:
    """The filing of this form (accession, filed) that first reported this period end. The HTML does not say."""
    gaap = companyfacts['facts']['us-gaap']
    best = None
    for metric, concept in concepts.items():
        for r in gaap.get(concept, {}).get('units', {}).get(METRICS[metric]['unit'], []):
            if r['end'] == report_date and r['form'] == form and (best is None or r['filed'] < best['filed']):
                best = r
    return best


def normalize_segments(parsed: dict[str, list[dict]], companyfacts: dict, ticker: str, form: str = '10-K') -> list[dict]:
    """
    parsed: {report date: parse_inline rows} per filing of this form. One facts row per (metric, axis, segment, fiscal
    year, fiscal period): 'FY' for 10-Ks, Q1-Q3 for 10-Qs, labelled by xbrl.period_label from the period's end date like
    the consolidated facts and the text chunks (a quarter's calendar year is not its fiscal year).

    The latest filing wins for a whole breakdown, not slice by slice. NVIDIA's FY2026 10-K re-presented FY2025 revenue
    by geography on a new basis and dropped Singapore; taking each slice from its latest filing kept the old Singapore
    row next to the new US/China/Taiwan rows, and the year summed to 154,181 against a 130,497 total. So each
    (metric, axis, fiscal year, fiscal period) comes entirely from the latest filing that reports it. A 10-Q also
    carries last year's quarter as a comparison, so a quarter's final figures come from the following year's 10-Q.
    """
    concepts = resolve(companyfacts, ticker).primary
    calendar = fiscal_calendar(companyfacts, ticker) if form == '10-Q' else None
    groups: dict[tuple, list[dict]] = {}
    for report_date in sorted(parsed):
        filing = _filing_for(companyfacts, concepts, report_date, form)
        if filing is None:
            continue
        this_filing: dict[tuple, list[dict]] = {}
        for f in parsed[report_date]:
            if not f['axis']:
                continue
            if calendar is None:
                fy, period = int(f['period_end'][:4]), 'FY'
            else:
                label = period_label(date.fromisoformat(f['period_end']), calendar)
                if label is None or label[1] == 'FY':
                    continue  # not a quarter end we can name
                fy, period = label
            this_filing.setdefault((f['metric'], f['axis'], fy, period), []).append({
                **f, 'ticker': ticker, 'fiscal_year': fy, 'fiscal_period': period, 'filing': report_date,
                'unit': METRICS[f['metric']]['unit'], 'derived': False, 'form': filing['form'],
                'accession': filing['accn'], 'filed': filing['filed'], 'split_factor': 1})
        groups |= this_filing  # replaces any breakdown an earlier filing gave for the same period
    rows = [r for g in groups.values() for r in g]
    return sorted(rows, key=lambda r: (r['metric'], r['axis'], r['fiscal_year'], r['fiscal_period'], r['segment']))


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
            balance = METRICS[f['metric']].get('balance')
            readings = {r['val'] for r in gaap.get(f['concept'], {}).get('units', {}).get('USD', [])
                        if r['end'] == f['period_end'] and (('start' not in r) if balance
                                                            else r.get('start') == f['period_start'])}
            if not readings:
                continue
            checked += 1
            if not any(abs(f['value'] - v) <= 1 for v in readings):
                bad.append({'metric': f['metric'], 'period_end': f['period_end'], 'parsed': f['value'],
                            'api': sorted(readings)})
    return checked, bad


def sum_check(rows: list[dict], consolidated: dict[tuple, float]) -> tuple[int, list[tuple]]:
    """
    Does each breakdown reconcile to the consolidated total? Slices can overlap (Data Center = Compute + Networking),
    so the children find_parts detects are left out and the remaining top-level slices must add up to the total within
    a million. An earlier version accepted any subset that matched, and it passed NVIDIA FY2025 geography with a stale
    Singapore row in it, because leaving that row out happened to match. consolidated maps (metric, fiscal_year) ->
    value. Returns (breakdowns checked, the (metric, axis, year) that did not reconcile).
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if not r['derived']:  # a remainder reconciles by construction, so it proves nothing
            groups.setdefault((r['metric'], r['axis'], r['fiscal_year']), []).append(r)

    checked, failed = 0, []
    for (metric, axis, fy), members in groups.items():
        total = consolidated.get((metric, fy))
        if total is None:
            continue
        checked += 1
        parts = find_parts(members)
        top = sum(round(r['value'] / 1e6) for r in members if (fy, r['segment']) not in parts)
        if abs(top - round(total / 1e6)) > SUM_TOLERANCE_M:
            failed.append((metric, axis, fy))
    return checked, failed


def match_items(target: float, items: list[dict]) -> list[dict] | None:
    """
    The one subset of corporate items (positive amounts, i.e. expenses) that adds up to target within a million, or None
    if no subset or more than one does. NVIDIA FY2026 tags six corporate items, including interest income and other
    non-operating income, and only stock compensation + unallocated expenses + acquisition costs make up its 8,910.
    """
    from itertools import combinations

    pool = [i for i in items if i['value'] > 0]
    if target <= 0 or not pool or len(pool) > MAX_SUBSET_MEMBERS:
        return None
    matches = [combo for size in range(1, len(pool) + 1) for combo in combinations(pool, size)
               if abs(sum(i['value'] for i in combo) - target) <= SUM_TOLERANCE_M * 1e6]
    return list(matches[0]) if len(matches) == 1 else None


def add_corporate_remainder(rows: list[dict], consolidated: dict[tuple, dict],
                            items: dict[tuple, list[dict]] | None = None, warnings: list[str] | None = None) -> list[dict]:
    """
    Companies keep some costs out of their segments (unallocated stock compensation and R&D), and not every filing tags
    that reconciling line as one figure: NVIDIA from FY2024 tags a corporate "total" equal to the segment total (dropped
    by parse_inline), Apple before FY2025 tags only part of it. For a business-segment breakdown with no reported
    corporate row, the gap is the consolidated total minus the segments. If the corporate items the same filing tags
    (items: (filing, start, end) -> parse_filing items) contain exactly one subset that adds up to the gap, the row is
    their sum, derived=False (every input is reported and the identity holds), with the items in `concept`. Otherwise
    the row is the gap itself, derived=True, citing the filing of the total; if items were tagged but none matched, a
    line goes to `warnings`. Business segments only: product and geography members overlap, so a remainder there means
    nothing. consolidated maps (metric, fiscal_year) -> the consolidated facts row the gap is measured against.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r['axis'] == 'segment':
            groups.setdefault((r['metric'], r['fiscal_year']), []).append(r)

    out = []
    for key, members in groups.items():
        total = consolidated.get(key)
        if total is None or any(m['segment'] == CORPORATE for m in members):
            continue
        remainder = total['value'] - sum(m['value'] for m in members)
        if abs(remainder) <= SUM_TOLERANCE_M * 1e6:
            continue  # the segments already add up
        first = members[0]
        tagged = (items or {}).get((first.get('filing'), first['period_start'], first['period_end']), [])
        match = match_items(-remainder, tagged)
        if match:
            out.append({**first, 'segment': CORPORATE, 'value': -sum(i['value'] for i in match),
                        'concept': '+'.join(i['concept'] for i in match), 'derived': False})
            continue
        if tagged and warnings is not None:
            listed = ', '.join(f"{i['concept']} {i['value'] / 1e6:,.0f}M" for i in tagged)
            warnings.append(f"{key[0]} {first['fiscal_period']} FY{key[1]}: corporate items tagged ({listed}) do not "
                            f"account for the gap of {remainder / 1e6:,.0f}M, derived instead")
        out.append({**first, 'segment': CORPORATE, 'value': remainder, 'concept': total['concept'],
                    'period_start': str(total['period_start']), 'period_end': str(total['period_end']),
                    'derived': True, 'form': total['form'], 'accession': total['accession'],
                    'filed': str(total['filed'])})
    return out


def choose_names(latest: dict[tuple, tuple[str, str]], labels: dict[str, dict]) -> dict[tuple, str]:
    """
    (axis, series key) -> display name. latest maps each series (keyed by member_name, which joins a member renamed
    between years) to (filing, member) of the latest filing that uses it; labels maps a filing to its parse_labels
    result. A series with no label keeps its key. If two series on one axis would get the same name, both keep their
    keys: the name is part of the facts table's primary key, so a clash would merge two series.
    """
    from earnings_rag.labels import display_name

    names = {}
    for (axis, key), (filing, member) in latest.items():
        name = display_name((labels.get(filing) or {}).get(member, {}))
        if name and key != CORPORATE:
            names[(axis, key)] = name
    seen: dict[tuple, list[tuple]] = {}
    for (axis, key), name in names.items():
        seen.setdefault((axis, name.lower()), []).append((axis, key))
    for clashing in (v for v in seen.values() if len(v) > 1):
        for k in clashing:
            names.pop(k)
    return names


def slice_names(ticker: str, companyfacts: dict, concepts: dict[str, str],
                parsed: dict[str, dict[str, list[dict]]]) -> dict[tuple, str]:
    """Fetch the label linkbase of the latest filing that uses each member (cached) and choose the names."""
    from earnings_rag.labels import fetch_labels

    latest: dict[tuple, tuple[str, str]] = {}   # (axis, key) -> (filing, member)
    forms = {}
    for form, by_date in parsed.items():
        for report_date, rows in by_date.items():
            forms[report_date] = form
            for r in rows:
                if r['axis'] and r.get('member') and latest.get((r['axis'], r['segment']), ('',))[0] <= report_date:
                    latest[(r['axis'], r['segment'])] = (report_date, r['member'])
    labels = {}
    for report_date in sorted({f for f, _ in latest.values()}):
        filing = _filing_for(companyfacts, concepts, report_date, forms[report_date])
        if filing is not None:
            labels[report_date] = fetch_labels(ticker, int(companyfacts['cik']), filing['accn'], report_date)
    names = choose_names(latest, labels)
    print(f'{ticker}: slice names from {len(labels)} label linkbases: {len(names)} of {len(latest)} slices named by '
          f'the filing, the rest keep their member names')
    return names


def ingest_segments(ticker: str, companyfacts: dict) -> bool:
    """
    Parse the cached 10-Ks and 10-Qs, validate, store. Returns False (storing nothing) if the parser disagrees with the
    API. Each period (FY, Q1-Q3) is checked and completed on its own: its slices must reconcile to that period's
    consolidated figure, and its corporate remainder is measured against it.
    """
    from earnings_rag import xbrl
    from earnings_rag.store import replace_segment_facts

    concepts = resolve(companyfacts, ticker).primary
    sources = {'10-K': (settings.raw_dir / ticker, ('FY',)), '10-Q': (settings.raw_dir / ticker / '10-Q', ('Q',))}
    parsed: dict[str, dict[str, list[dict]]] = {}
    items: dict[tuple, list[dict]] = {}   # (filing, period start, period end) -> corporate items
    for form, (folder, durations) in sources.items():
        parsed[form] = {}
        for p in sorted(folder.glob('*.html')):
            parsed[form][p.stem], tagged = parse_filing(p.read_bytes(), concepts, durations)
            for i in tagged:
                items.setdefault((p.stem, i['period_start'], i['period_end']), []).append(i)
    checked, bad = cross_check({f'{form}/{d}': rows for form, by_date in parsed.items() for d, rows in by_date.items()},
                               companyfacts, ticker)
    print(f'{ticker}: segment parser check: {checked} consolidated values vs the SEC API, {len(bad)} mismatches')
    for b in bad[:5]:
        print(f'  mismatch {b}')
    if bad:
        print(f'{ticker}: segment parser disagrees with the API, segments not stored')
        return False

    rows = [r for form, by_date in parsed.items() for r in normalize_segments(by_date, companyfacts, ticker, form)]
    consolidated = xbrl.normalize(companyfacts, ticker)
    # Each breakdown ends in one of four outcomes: it reconciles as reported (incl. a reported corporate figure), via
    # the corporate items the filing tags, via a derived remainder, or not at all (a real problem worth reading)
    corporate, warnings, checked_axes = [], [], 0
    outcomes: dict[str, list[str]] = {'items': [], 'derived': [], 'failed': []}
    for period in ('FY', 'Q1', 'Q2', 'Q3'):
        totals = {(r['metric'], r['fiscal_year']): r for r in consolidated if r['fiscal_period'] == period}
        period_rows = [r for r in rows if r['fiscal_period'] == period]
        axes_checked, failed = sum_check(period_rows, {k: r['value'] for k, r in totals.items()})
        checked_axes += axes_checked
        added = add_corporate_remainder(period_rows, totals, items, warnings)
        corporate += added
        closed = {(r['metric'], r['axis'], r['fiscal_year']): r['derived'] for r in added}
        for metric, axis, fy in failed:
            kind = 'failed' if (metric, axis, fy) not in closed else 'derived' if closed[(metric, axis, fy)] else 'items'
            outcomes[kind].append(f'{metric} by {axis} {period} FY{fy}')
    as_reported = checked_axes - sum(len(v) for v in outcomes.values())
    print(f'{ticker}: {len(rows)} segment facts; of {checked_axes} breakdowns {as_reported} reconcile as reported, '
          f'{len(outcomes["items"])} via reported corporate items, {len(outcomes["derived"])} via a derived remainder, '
          f'{len(outcomes["failed"])} do not reconcile')
    for label, key in (('derived remainder', 'derived'), ('DOES NOT RECONCILE', 'failed')):
        if outcomes[key]:
            print(f'  {label}: ' + '; '.join(outcomes[key]))
    for w in warnings:
        print(f'  {w}')
    names = slice_names(ticker, companyfacts, concepts, parsed)
    for r in rows + corporate:
        r['segment'] = names.get((r['axis'], r['segment']), r['segment'])
    replace_segment_facts(ticker, rows + corporate)
    return True


def find_parts(rows: list[dict]) -> dict[tuple[int, str], str]:
    """
    Slices on one axis can overlap: NVIDIA's Data Center is Compute + Networking, Apple's Products is iPhone + Mac +
    iPad + Wearables. Adding all of them double-counts. A slice that equals the sum of two or more other slices in the
    same year is their parent. Returns (fiscal_year, child) -> parent. Detected from the numbers, so no per-company list.
    """
    from itertools import combinations

    parts = {}
    years: dict[int, list[tuple[str, int]]] = {}
    for r in rows:
        if not r['derived'] and r['segment'] != CORPORATE and r['value'] > 0:
            years.setdefault(r['fiscal_year'], []).append((r['segment'], round(r['value'] / 1e6)))

    for fy, members in years.items():
        if len(members) > MAX_SUBSET_MEMBERS:
            continue
        for parent, total in sorted(members, key=lambda m: -m[1]):
            others = [m for m in members if m[0] != parent and (fy, m[0]) not in parts and m[1] < total]
            match = next((combo for size in range(2, len(others) + 1) for combo in combinations(others, size)
                          if abs(sum(v for _, v in combo) - total) <= SUM_TOLERANCE_M), None)
            for child, _ in match or ():
                parts[(fy, child)] = parent
    return parts
