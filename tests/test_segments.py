from datetime import date

from earnings_rag import segments
from earnings_rag.segments import (parse_inline, member_name, normalize_segments, cross_check, sum_check,
                                   add_corporate_remainder, CORPORATE)

M = 1_000_000


def ixbrl(contexts, facts):
    """contexts: {id: (start, end, [(axis, member)])}, facts: [(concept, context id, text, extra attributes)]."""
    ctx = ''
    for cid, (start, end, dims) in contexts.items():
        members = ''.join(f'<xbrldi:explicitMember dimension="{a}">{m}</xbrldi:explicitMember>' for a, m in dims)
        seg = f'<xbrli:entity><xbrli:segment>{members}</xbrli:segment></xbrli:entity>' if dims else ''
        ctx += (f'<xbrli:context id="{cid}">{seg}<xbrli:period><xbrli:startDate>{start}</xbrli:startDate>'
                f'<xbrli:endDate>{end}</xbrli:endDate></xbrli:period></xbrli:context>')
    body = ''.join(f'<ix:nonFraction name="us-gaap:{c}" contextRef="{cid}" {attrs}>{text}</ix:nonFraction>'
                   for c, cid, text, attrs in facts)
    return (f'<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" xmlns:xbrli="http://www.xbrl.org/2003/instance" '
            f'xmlns:xbrldi="http://xbrl.org/2006/xbrldi"><body><ix:header><ix:resources>{ctx}</ix:resources></ix:header>'
            f'{body}</body></html>').encode()


FY = ('2024-01-29', '2025-01-26')
PRODUCT = 'srt:ProductOrServiceAxis'
SEGMENT = 'us-gaap:StatementBusinessSegmentsAxis'
ITEMS = 'srt:ConsolidationItemsAxis'


def by_key(rows):
    return {(r['metric'], r['axis'], r['segment']): r['value'] for r in rows}


def test_scale_sign_and_fixed_zero_are_applied():
    html = ixbrl({'a': (*FY, [(PRODUCT, 'nvda:DataCenterMember')]), 'b': (*FY, [(PRODUCT, 'nvda:GamingMember')]),
                  'c': (*FY, [(PRODUCT, 'nvda:AutomotiveMember')]), 'd': (*FY, [(PRODUCT, 'nvda:OEMAndOtherMember')])},
                 [('Revenues', 'a', '115,186', 'scale="6"'), ('Revenues', 'b', '1.5', 'scale="9"'),
                  ('Revenues', 'c', '—', 'format="ixt:fixed-zero" scale="6"'),
                  ('Revenues', 'd', '389', 'scale="6" sign="-"')])
    got = by_key(parse_inline(html, 'NVDA'))
    assert got[('revenue', 'product', 'Data Center')] == 115_186 * M
    assert got[('revenue', 'product', 'Gaming')] == 1_500 * M
    assert got[('revenue', 'product', 'Automotive')] == 0
    assert got[('revenue', 'product', 'OEM and Other')] == -389 * M


def test_segment_with_and_without_the_operating_segments_qualifier_is_the_same_series():
    old = ixbrl({'a': (*FY, [(SEGMENT, 'aapl:AmericasSegmentMember')])},
                [('RevenueFromContractWithCustomerExcludingAssessedTax', 'a', '100', 'scale="6"')])
    new = ixbrl({'a': (*FY, [(ITEMS, 'us-gaap:OperatingSegmentsMember'), (SEGMENT, 'aapl:AmericasSegmentMember')])},
                [('RevenueFromContractWithCustomerExcludingAssessedTax', 'a', '100', 'scale="6"')])
    assert parse_inline(old, 'AAPL') == parse_inline(new, 'AAPL')
    assert parse_inline(new, 'AAPL')[0]['segment'] == 'Americas'


def test_renamed_member_gives_the_same_name():
    assert member_name('nvda:ComputeAndNetworkingMember') == member_name('nvda:ComputeAndNetworkingSegmentMember')
    assert member_name('nvda:ComputeAndNetworkingMember') == 'Compute and Networking'
    assert member_name('country:US') == 'US'
    assert member_name('us-gaap:CorporateNonSegmentMember') == CORPORATE


def test_contexts_with_several_dimensions_or_other_axes_are_skipped():
    html = ixbrl({'a': (*FY, [(ITEMS, 'us-gaap:OperatingSegmentsMember'), (PRODUCT, 'cof:FeesMember'),
                              (SEGMENT, 'cof:CreditCardSegmentMember')]),
                  'b': (*FY, [('us-gaap:StatementEquityComponentsAxis', 'us-gaap:RetainedEarningsMember')]),
                  'c': (*FY, [(ITEMS, 'us-gaap:OperatingSegmentsMember')]),   # total of the segments
                  'd': ('2025-01-26', '2025-01-26', [])},                      # instant
                [('Revenues', c, '5', 'scale="6"') for c in 'abcd'])
    assert parse_inline(html, 'COF') == []


def test_quarterly_and_unmapped_concepts_are_ignored():
    html = ixbrl({'q': ('2024-10-28', '2025-01-26', []), 'y': (*FY, [])},
                 [('Revenues', 'q', '5', 'scale="6"'), ('SomethingElse', 'y', '5', 'scale="6"')])
    assert parse_inline(html, 'NVDA') == []


def test_a_corporate_item_that_repeats_the_segment_total_is_dropped():
    html = ixbrl({'t': (*FY, [(ITEMS, 'us-gaap:OperatingSegmentsMember')]),
                  'c': (*FY, [(ITEMS, 'us-gaap:CorporateNonSegmentMember')])},
                 [('OperatingIncomeLoss', 't', '87,960', 'scale="6"'), ('OperatingIncomeLoss', 'c', '87,960', 'scale="6"')])
    assert parse_inline(html, 'NVDA') == []


def companyfacts(*rows):
    return {'facts': {'us-gaap': {'Revenues': {'units': {'USD': list(rows)}}}}}


def api(val, end='2025-01-26', filed='2025-02-26', start='2024-01-29', form='10-K'):
    return {'start': start, 'end': end, 'val': val, 'accn': f'acc-{filed}', 'form': form, 'filed': filed}


def test_cross_check_catches_a_scale_error_in_the_parser():
    right = ixbrl({'a': (*FY, [])}, [('Revenues', 'a', '130,497', 'scale="6"')])
    wrong = ixbrl({'a': (*FY, [])}, [('Revenues', 'a', '130,497', 'scale="3"')])
    facts = companyfacts(api(130_497 * M))
    assert cross_check({'x': parse_inline(right, 'NVDA')}, facts, 'NVDA') == (1, [])
    checked, bad = cross_check({'x': parse_inline(wrong, 'NVDA')}, facts, 'NVDA')
    assert checked == 1 and len(bad) == 1


def test_latest_filing_wins_and_cites_its_own_accession():
    def filing(dc):
        return [{'metric': 'revenue', 'concept': 'Revenues', 'axis': 'product', 'segment': 'Data Center', 'value': dc,
                 'period_start': FY[0], 'period_end': FY[1]}]
    facts = companyfacts(api(1, filed='2025-02-26'), api(1, end='2026-01-25', start='2025-01-27', filed='2026-02-25'))
    parsed = {'2025-01-26': filing(100 * M), '2026-01-25': filing(110 * M)}
    rows = normalize_segments(parsed, facts, 'NVDA')
    assert len(rows) == 1 and rows[0]['value'] == 110 * M          # restated figure wins
    assert rows[0]['accession'] == 'acc-2026-02-25' and rows[0]['filed'] == '2026-02-25'
    assert rows[0]['fiscal_year'] == 2025 and rows[0]['fiscal_period'] == 'FY' and rows[0]['segment'] == 'Data Center'


def row(metric, axis, segment, value, fy=2025):
    return {'metric': metric, 'axis': axis, 'segment': segment, 'value': value * M, 'fiscal_year': fy,
            'concept': 'X', 'period_start': '2024-01-29', 'period_end': '2025-01-26', 'derived': False, 'form': '10-K',
            'accession': 'seg', 'filed': '2025-02-26', 'unit': 'USD', 'ticker': 'NVDA', 'fiscal_period': 'FY',
            'split_factor': 1}


def test_sum_check_handles_a_hierarchy_and_flags_a_breakdown_that_does_not_add_up():
    total = {('revenue', 2025): 130_497 * M}
    products = [row('revenue', 'product', n, v) for n, v in
                [('Data Center', 115_186), ('Compute', 102_196), ('Networking', 12_990),   # Data Center = Compute + Networking
                 ('Gaming', 11_350), ('Pro Viz', 1_878), ('Auto', 1_694), ('OEM', 389)]]
    assert sum_check(products, total) == (1, [])
    broken = [row('revenue', 'geography', 'US', 61_257), row('revenue', 'geography', 'TW', 20_573)]
    assert sum_check(broken, total) == (1, [('revenue', 'geography', 2025)])


TOTAL = {('operating_income', 2025): {'value': 130_387 * M, 'concept': 'OperatingIncomeLoss', 'form': '10-K',
                                      'accession': 'total-acc', 'filed': '2026-02-25', 'period_start': date(2024, 1, 29),
                                      'period_end': date(2025, 1, 26)}}


def test_missing_corporate_item_is_derived_as_total_minus_segments():
    rows = [row('operating_income', 'segment', 'Compute and Networking', 130_141),
            row('operating_income', 'segment', 'Graphics', 9_156)]
    [r] = add_corporate_remainder(rows, TOTAL)
    assert r['segment'] == CORPORATE and r['axis'] == 'segment'
    assert r['value'] == -8_910 * M and r['derived']
    assert r['accession'] == 'total-acc'            # cites the filing that reported the total
    assert sum_check(rows + [r], {('operating_income', 2025): 130_387 * M}) == (1, [('operating_income', 'segment', 2025)])


def test_no_remainder_when_corporate_is_reported_or_the_segments_already_add_up():
    reported = [row('operating_income', 'segment', 'Compute and Networking', 138_841),
                row('operating_income', 'segment', CORPORATE, -8_454)]
    assert add_corporate_remainder(reported, TOTAL) == []
    adds_up = [row('operating_income', 'segment', 'A', 100_000), row('operating_income', 'segment', 'B', 30_387)]
    assert add_corporate_remainder(adds_up, TOTAL) == []


def test_no_remainder_for_overlapping_axes_or_years_without_a_consolidated_total():
    overlapping = [row('operating_income', 'product', 'Data Center', 1)]
    assert add_corporate_remainder(overlapping, TOTAL) == []
    other_year = [row('operating_income', 'segment', 'A', 1, fy=2030)]
    assert add_corporate_remainder(other_year, TOTAL) == []


def test_ingest_replaces_the_tickers_breakdown_rows_including_the_derived_remainder(monkeypatch, tmp_path):
    (tmp_path / 'NVDA').mkdir()
    html = ixbrl({'t': (*FY, []), 'a': (*FY, [(ITEMS, 'us-gaap:OperatingSegmentsMember'),
                                              (SEGMENT, 'nvda:ComputeAndNetworkingSegmentMember')])},
                 [('OperatingIncomeLoss', 't', '90', 'scale="6"'), ('OperatingIncomeLoss', 'a', '100', 'scale="6"')])
    (tmp_path / 'NVDA' / '2025-01-26.html').write_bytes(html)
    facts = {'facts': {'us-gaap': {'OperatingIncomeLoss': {'units': {'USD': [api(90 * M)]}}}}}

    stored = {}
    monkeypatch.setattr(segments, 'settings', type('S', (), {'raw_dir': tmp_path})())
    monkeypatch.setattr('earnings_rag.store.replace_segment_facts', lambda t, rows: stored.update({t: rows}))
    assert segments.ingest_segments('NVDA', facts)
    assert [(r['segment'], r['value'] / M, r['derived']) for r in stored['NVDA']] == [
        ('Compute and Networking', 100, False), (CORPORATE, -10, True)]


def test_ingest_stores_nothing_when_the_parser_disagrees_with_the_api(monkeypatch, tmp_path):
    (tmp_path / 'NVDA').mkdir()
    (tmp_path / 'NVDA' / '2025-01-26.html').write_bytes(ixbrl({'t': (*FY, [])}, [('OperatingIncomeLoss', 't', '90', 'scale="3"')]))
    facts = {'facts': {'us-gaap': {'OperatingIncomeLoss': {'units': {'USD': [api(90 * M)]}}}}}
    stored = []
    monkeypatch.setattr(segments, 'settings', type('S', (), {'raw_dir': tmp_path})())
    monkeypatch.setattr('earnings_rag.store.replace_segment_facts', lambda t, rows: stored.append(rows))
    assert not segments.ingest_segments('NVDA', facts)
    assert stored == []


def test_find_parts_detects_children_from_the_numbers():
    rows = [row('revenue', 'product', n, v) for n, v in
            [('Products', 307_003), ('Services', 109_158), ('iPhone', 209_586), ('Mac', 33_708), ('iPad', 28_023),
             ('Wearables', 35_686)]]
    parts = segments.find_parts(rows)
    assert parts == {(2025, c): 'Products' for c in ('iPhone', 'Mac', 'iPad', 'Wearables')}
    flat = [row('revenue', 'geography', n, v) for n, v in [('US', 151_790), ('CN', 64_377), ('Other', 199_994)]]
    assert segments.find_parts(flat) == {}


def test_a_later_filing_replaces_the_whole_breakdown_for_a_year():
    # NVIDIA's FY2026 10-K re-presented FY2025 geography and dropped Singapore; the old Singapore row must not survive
    def geo(name, value):
        return {'metric': 'revenue', 'concept': 'Revenues', 'axis': 'geography', 'segment': name, 'value': value * M,
                'period_start': FY[0], 'period_end': FY[1]}
    facts = companyfacts(api(1, filed='2025-02-26'), api(1, end='2026-01-25', start='2025-01-27', filed='2026-02-25'))
    parsed = {'2025-01-26': [geo('US', 61_257), geo('SG', 23_684)], '2026-01-25': [geo('US', 77_482)]}
    rows = normalize_segments(parsed, facts, 'NVDA')
    assert [(r['segment'], r['value'] / M) for r in rows] == [('US', 77_482)]


def test_sum_check_is_not_fooled_by_a_subset_that_happens_to_match():
    total = {('revenue', 2025): 130_497 * M}
    stale = [row('revenue', 'geography', n, v) for n, v in
             [('US', 77_482), ('China', 25_048), ('TW', 23_600), ('Other', 4_367), ('SG', 23_684)]]
    assert sum_check(stale, total) == (1, [('revenue', 'geography', 2025)])


def test_balances_are_read_only_at_a_fiscal_year_end():
    segment = [(ITEMS, 'us-gaap:OperatingSegmentsMember'), (SEGMENT, 'cof:ConsumerBankingSegmentMember')]
    html = ixbrl({'fy': ('2024-01-01', '2024-12-31', []), 'ye': ('2024-12-31', '2024-12-31', segment),
                  'mid': ('2024-06-30', '2024-06-30', segment)},
                 [('Deposits', 'ye', '318,329', 'scale="6"'), ('Deposits', 'mid', '1', 'scale="6"')])
    html = html.replace(b'<xbrli:startDate>2024-12-31</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate>',
                        b'<xbrli:instant>2024-12-31</xbrli:instant>')
    html = html.replace(b'<xbrli:startDate>2024-06-30</xbrli:startDate><xbrli:endDate>2024-06-30</xbrli:endDate>',
                        b'<xbrli:instant>2024-06-30</xbrli:instant>')
    [r] = parse_inline(html, 'COF')
    assert (r['metric'], r['segment'], r['value'], r['period_end']) == ('deposits', 'Consumer Banking', 318_329 * M,
                                                                       '2024-12-31')


def test_cross_check_compares_a_balance_with_the_apis_date_only_value():
    html = ixbrl({'fy': ('2024-01-01', '2024-12-31', []), 'ye': ('2024-12-31', '2024-12-31', [])},
                 [('Deposits', 'ye', '362,707', 'scale="6"')]).replace(
        b'<xbrli:startDate>2024-12-31</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate>',
        b'<xbrli:instant>2024-12-31</xbrli:instant>')
    facts = {'facts': {'us-gaap': {'Deposits': {'units': {'USD': [
        {'end': '2024-12-31', 'val': 362_707 * M, 'accn': 'a', 'form': '10-K', 'filed': '2025-02-20'}]}}}}}
    assert cross_check({'x': parse_inline(html, 'COF')}, facts, 'COF') == (1, [])
