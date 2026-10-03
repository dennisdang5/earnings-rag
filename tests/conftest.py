import pytest

from earnings_rag.agent import loop


@pytest.fixture(autouse=True)
def no_filings_in_the_prompt(monkeypatch):
    """run_agent builds the date context from the database by default; CI has none when pytest runs."""
    monkeypatch.setattr(loop, 'text_periods', lambda: [])
