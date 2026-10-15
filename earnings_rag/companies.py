"""
The home screen's company cards: for each ticker, its name, newest annual and quarterly filing with text, and its latest
fiscal-year revenue. `build_cards` is pure (filings and the revenue lookup are passed in) so tests need no database.
"""
from typing import Callable


def _filing(filings: list[dict], ticker: str, form: str) -> dict | None:
    """The newest filing of this form for the ticker, as {label, period} (label like 'FY2026' or 'Q2 FY2027')."""
    rows = [f for f in filings if f['ticker'] == ticker and f['form'] == form]
    if not rows:
        return None
    newest = max(rows, key=lambda f: f['period'])
    quarter = '' if newest['fiscal_period'] == 'FY' else f"{newest['fiscal_period']} "
    return {'label': f"{quarter}FY{newest['fiscal_year']}", 'period': str(newest['period'])}


def build_cards(tickers: list[str], names: dict[str, str], filings: list[dict],
                latest_revenue: Callable[[str], dict | None]) -> list[dict]:
    """One card per ticker, in the configured order. `latest_revenue(ticker)` returns {'fiscal_year', 'usd_millions'} or None."""
    return [{
        'ticker': t,
        'name': names.get(t, t),
        'latest_annual': _filing(filings, t, '10-K'),
        'latest_quarterly': _filing(filings, t, '10-Q'),
        'revenue': latest_revenue(t),
    } for t in tickers]
