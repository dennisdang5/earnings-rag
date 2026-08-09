import requests
from earnings_rag.config import settings

def _sec_headers() -> dict[str, str]:
    if not settings.sec_user_agent:
        raise RuntimeError('SEC_USER_AGENT not set - EDGAR will return 403')
    return {'User-Agent': settings.sec_user_agent}

SEC_HEADERS = _sec_headers()
def load_ticker_map() -> dict[str, str]:
    r = requests.get('https://www.sec.gov/files/company_tickers.json',
                     headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    ticker_map = {}
    for row in data.values():
        ticker = row['ticker']
        cik_raw = row['cik_str'] # Central index key a unique code assigned by SEC to identify corporations
        cik = str(cik_raw).zfill(10)
        ticker_map[ticker] = cik

    return ticker_map

tm = load_ticker_map()
print(len(tm), tm['NVDA'], tm['AAPL'])