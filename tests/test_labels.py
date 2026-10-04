from earnings_rag.labels import display_name, parse_labels
from earnings_rag.segments import choose_names

LAB = '''<link:linkbase xmlns:link="http://www.xbrl.org/2003/linkbase" xmlns:xlink="http://www.w3.org/1999/xlink">
<link:labelLink>
<link:loc xlink:href="aapl-20250927.xsd#aapl_IPhoneMember" xlink:label="loc_iphone"/>
<link:label xlink:label="lab_iphone" xlink:role="http://www.xbrl.org/2003/role/label">iPhone [Member]</link:label>
<link:label xlink:label="lab_iphone" xlink:role="http://www.xbrl.org/2003/role/terseLabel">iPhone</link:label>
<link:labelArc xlink:from="loc_iphone" xlink:to="lab_iphone"/>
<link:loc xlink:href="https://xbrl.fasb.org/country.xsd#country_US" xlink:label="loc_us"/>
<link:label xlink:label="lab_us" xlink:role="http://www.xbrl.org/2003/role/label">UNITED STATES</link:label>
<link:labelArc xlink:from="loc_us" xlink:to="lab_us"/>
</link:labelLink></link:linkbase>'''


def test_parse_labels_maps_concepts_to_their_labels_by_role():
    labels = parse_labels(LAB)
    assert labels['aapl:IPhoneMember'] == {'label': 'iPhone [Member]', 'terseLabel': 'iPhone'}
    assert labels['country:US'] == {'label': 'UNITED STATES'}


def test_display_name_prefers_the_terse_label_the_tables_print():
    assert display_name({'label': 'iPhone [Member]', 'terseLabel': 'iPhone'}) == 'iPhone'
    assert display_name({'label': 'Compute And Networking Segment [Member]',
                         'terseLabel': 'Compute & Networking'}) == 'Compute & Networking'
    assert display_name({'label': 'UNITED STATES', 'terseLabel': 'U.S.'}) == 'U.S.'


def test_a_terse_label_that_is_only_a_piece_of_the_standard_one_is_not_used():
    # NVIDIA's OtherCountriesMember and Capital One's OtherContractRevenueMember are both tersely "Other"
    assert display_name({'label': 'Other Countries [Member]', 'terseLabel': 'Other'}) == 'Other Countries'
    assert display_name({'label': 'Other Countries [Member]', 'terseLabel': 'Other countries'}) == 'Other countries'


def test_without_a_terse_label_the_standard_one_is_cleaned():
    assert display_name({'label': 'UNITED STATES'}) == 'United States'
    assert display_name({'label': 'Graphics Segment [Member]'}) == 'Graphics'
    assert display_name({}) is None


def test_names_come_from_the_latest_filing_and_a_clash_keeps_the_member_names():
    latest = {('product', 'I Phone'): ('2025-09-27', 'aapl:IPhoneMember'),
              ('geography', 'Other Countries'): ('2026-01-25', 'nvda:OtherCountriesMember'),
              ('geography', 'All Other Countries'): ('2026-01-25', 'nvda:AllOtherMember'),
              ('product', 'Retired'): ('2022-09-24', 'aapl:RetiredMember')}
    labels = {'2025-09-27': {'aapl:IPhoneMember': {'terseLabel': 'iPhone'}},
              '2026-01-25': {'nvda:OtherCountriesMember': {'terseLabel': 'Other'},
                             'nvda:AllOtherMember': {'terseLabel': 'Other'}},
              '2022-09-24': None}                                        # no linkbase: the member name stays
    assert choose_names(latest, labels) == {('product', 'I Phone'): 'iPhone'}
