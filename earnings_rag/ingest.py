import requests
from earnings_rag.config import settings

def _sec_headers() -> dict[str, str]:
    if not settings.sec_user_agent:
        raise RuntimeError('SEC_USER_AGENT not set - EDGAR will return 403')
    return {'User-Agent': settings.sec_user_agent}

session = requests.Session()
session.headers.update(_sec_headers())

def load_ticker_map() -> dict[str, str]:
    r = session.get('https://www.sec.gov/files/company_tickers.json', timeout=30)
    r.raise_for_status()
    data = r.json()

    ticker_map = {}
    for row in data.values():
        ticker = row['ticker']
        cik_raw = row['cik_str'] # Central index key a unique code assigned by SEC to identify corporations
        cik = str(cik_raw).zfill(10)
        ticker_map[ticker] = cik

    return ticker_map

def get_filings(cik: str, form_type: str = '10-K') -> list[dict]:
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    r = session.get(url, timeout=30)
    r.raise_for_status()

    data = r.json()
    recent = data['filings']['recent']

    accessions = recent['accessionNumber']
    forms = recent['form']
    dates = recent['filingDate']
    docs = recent['primaryDocument']

    results = []
    for i in range(len(forms)):
        if forms[i] != form_type:
            continue
        results.append(({
            'accession': accessions[i],
            'form': forms[i],
            'date': dates[i],
            'doc': docs[i]
        }))

    return results


if __name__ == '__main__':
    tm = load_ticker_map()
    filings = get_filings(tm['NVDA'], '10-K')
    for f in filings:
        print(f['date'], f['accession'], f['doc'])