from earnings_rag.companies import build_cards


def filing(ticker, form, period, fy, fp):
    return {'ticker': ticker, 'form': form, 'period': period, 'fiscal_year': fy, 'fiscal_period': fp}


FILINGS = [
    filing('NVDA', '10-K', '2025-01-26', 2025, 'FY'), filing('NVDA', '10-K', '2026-01-25', 2026, 'FY'),
    filing('NVDA', '10-Q', '2026-04-26', 2027, 'Q1'), filing('NVDA', '10-Q', '2026-07-26', 2027, 'Q2'),
    filing('AAPL', '10-K', '2025-09-27', 2025, 'FY'),
]


def test_cards_pick_each_companys_newest_filings_in_configured_order():
    cards = build_cards(['AAPL', 'NVDA'], {'NVDA': 'NVIDIA'}, FILINGS,
                        lambda t: {'fiscal_year': 2026, 'usd_millions': 215938} if t == 'NVDA' else None)
    assert [c['ticker'] for c in cards] == ['AAPL', 'NVDA']
    nvda = cards[1]
    assert nvda['name'] == 'NVIDIA'
    assert nvda['latest_annual'] == {'label': 'FY2026', 'period': '2026-01-25'}
    assert nvda['latest_quarterly'] == {'label': 'Q2 FY2027', 'period': '2026-07-26'}
    assert nvda['revenue'] == {'fiscal_year': 2026, 'usd_millions': 215938}


def test_a_company_without_quarterlies_or_revenue_or_a_name_still_gets_a_card():
    apple = build_cards(['AAPL'], {}, FILINGS, lambda t: None)[0]
    assert apple['name'] == 'AAPL' and apple['latest_quarterly'] is None and apple['revenue'] is None
    assert build_cards(['TSLA'], {}, FILINGS, lambda t: None)[0]['latest_annual'] is None


def test_companies_endpoint_returns_the_cards(monkeypatch):
    from fastapi.testclient import TestClient
    from earnings_rag import api

    monkeypatch.setattr(api, 'get_company_cards', lambda: [{'ticker': 'NVDA', 'name': 'NVIDIA'}])
    response = TestClient(api.app).get('/companies')
    assert response.status_code == 200 and response.json() == [{'ticker': 'NVDA', 'name': 'NVIDIA'}]
