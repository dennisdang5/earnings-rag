import requests
from earnings_rag.config import settings
import time
from pathlib import Path

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
    report_dates = recent['reportDate'] # Fiscal period end

    results = []
    for i in range(len(forms)):
        if forms[i] != form_type:
            continue
        results.append(({
            'accession': accessions[i],
            'form': forms[i],
            'date': dates[i],
            'doc': docs[i],
            'report_date': report_dates[i]
        }))

    return results

def filing_url(cik: str, accession: str, doc:str) -> str:
    cik_bare = cik.lstrip('0')
    acc_bare = accession.replace('-', '')
    return f'https://www.sec.gov/Archives/edgar/data/{cik_bare}/{acc_bare}/{doc}'

DATA_RAW = Path('data/raw')
def download_filing(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    r = session.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    time.sleep(0.15)
    return dest


TICKERS = ['NVDA', 'AAPL', 'COF']

def ingest_all(n_per_ticker: int = 4) -> None:
    tm = load_ticker_map()
    for ticker in TICKERS:
        cik = tm[ticker]
        for f in get_filings(cik, '10-K')[:n_per_ticker]:
            url = filing_url(cik, f['accession'], f['doc'])
            dest = DATA_RAW / ticker / f"{f['report_date']}.html"
            download_filing(url, dest)
            print(f"{ticker} {f['report_date']} -> {dest}")

if __name__ == '__main__':
    ingest_all()