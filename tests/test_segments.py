from earnings_rag.segments import (parse_inline, member_name, normalize_segments, cross_check, sum_check, CORPORATE)

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
    return {'metric': metric, 'axis': axis, 'segment': segment, 'value': value * M, 'fiscal_year': fy}


def test_sum_check_handles_a_hierarchy_and_flags_a_breakdown_that_does_not_add_up():
    total = {('revenue', 2025): 130_497 * M}
    products = [row('revenue', 'product', n, v) for n, v in
                [('Data Center', 115_186), ('Compute', 102_196), ('Networking', 12_990),   # Data Center = Compute + Networking
                 ('Gaming', 11_350), ('Pro Viz', 1_878), ('Auto', 1_694), ('OEM', 389)]]
    assert sum_check(products, total) == (1, [])
    broken = [row('revenue', 'geography', 'US', 61_257), row('revenue', 'geography', 'TW', 20_573)]
    assert sum_check(broken, total) == (1, [('revenue', 'geography', 2025)])
