import re
import time

from bs4 import BeautifulSoup

from earnings_rag.config import settings

ROLE = 'http://www.xbrl.org/2003/role/'


def parse_labels(xml: str) -> dict[str, dict[str, str]]:
    """
    A label linkbase as {concept: {role: text}}, concepts written prefix:Name ('aapl:IPhoneMember', 'country:US') and
    roles by their short name ('label', 'terseLabel'). Pure: no network.
    """
    soup = BeautifulSoup(xml, 'xml')
    locs = {}
    for loc in soup.find_all('loc'):
        fragment = loc['xlink:href'].split('#')[-1]          # aapl_IPhoneMember
        locs[loc['xlink:label']] = fragment.replace('_', ':', 1)
    texts: dict[str, dict[str, str]] = {}
    for label in soup.find_all('label'):
        if label.get('xlink:label'):
            role = label.get('xlink:role', ROLE + 'label').replace(ROLE, '')
            texts.setdefault(label['xlink:label'], {})[role] = label.get_text(strip=True)
    out: dict[str, dict[str, str]] = {}
    for arc in soup.find_all('labelArc'):
        concept = locs.get(arc['xlink:from'])
        if concept:
            out.setdefault(concept, {}).update(texts.get(arc['xlink:to'], {}))
    return out


def _words(text: str) -> set[str]:
    return set(re.findall(r'[a-z0-9]+', text.lower().replace('&', ' and ')))  # "Compute & Networking" is not shorter


def display_name(labels: dict[str, str]) -> str | None:
    """
    The name a slice is shown under, from one member's labels. The terse label is what the filing's tables print
    ("iPhone", "Compute & Networking", "U.S."), unless it is missing or only a shorter piece of the standard label:
    NVIDIA's OtherCountriesMember and Capital One's OtherContractRevenueMember are both tersely "Other", which says
    nothing on its own. Then the standard label, without "[Member]" and "Segment", and country names (all capitals in
    the country taxonomy, "UNITED STATES") in title case.
    """
    standard = labels.get('label')
    if standard:
        standard = re.sub(r'\s*\[Member\]$', '', standard).strip()
        standard = re.sub(r'\s+Segment$', '', standard)
        if standard.isupper():
            standard = standard.title()
    terse = labels.get('terseLabel')
    if terse and not (standard and _words(terse) < _words(standard)):
        return terse
    return standard or None


def fetch_labels(ticker: str, cik: int, accession: str, report_date: str, refresh: bool = False) -> dict | None:
    """
    The parsed label linkbase of one filing, cached as data/xbrl/labels/<TICKER>_<report date>_lab.xml. None if the
    filing has none or it cannot be fetched (names then fall back to segments.member_name).
    """
    path = settings.data_dir / 'xbrl' / 'labels' / f'{ticker}_{report_date}_lab.xml'
    if path.exists() and not refresh:
        return parse_labels(path.read_text(encoding='utf-8'))

    from earnings_rag.ingest import session  # raises at import without SEC_USER_AGENT, so not at the top

    base = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace("-", "")}/'
    try:
        names = [i['name'] for i in session.get(base + 'index.json', timeout=30).json()['directory']['item']]
        # most filings ship a separate _lab.xml; some embed the linkbases in the .xsd schema
        for name in [n for n in names if n.endswith('_lab.xml')] or [n for n in names if n.endswith('.xsd')]:
            text = session.get(base + name, timeout=60).text
            time.sleep(0.15)
            if 'labelLink' in text:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding='utf-8')
                return parse_labels(text)
    except Exception as e:  # a name is cosmetic: never fail ingestion over it
        print(f'{ticker}: labels for {report_date} not fetched ({e})')
    return None
