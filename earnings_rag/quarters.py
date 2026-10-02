from datetime import date

# Pure helpers for quarterly ingestion, kept apart from ingest.py because that module raises at import when
# SEC_USER_AGENT is unset (as in CI), and these must be testable there.


def window_start(oldest_10k_period: str) -> date:
    """One year before the oldest ingested 10-K's fiscal year end: the point after which 10-Qs belong to its fiscal years."""
    d = date.fromisoformat(oldest_10k_period)
    try:
        return d.replace(year=d.year - 1)
    except ValueError:  # Feb 29
        return d.replace(year=d.year - 1, day=28)


def select_quarters(filings: list[dict], oldest_10k_period: str) -> list[dict]:
    """
    The 10-Qs worth ingesting: every one whose period ends after the start of the oldest 10-K's fiscal year (the quarters
    that line up with the ingested years) up to the latest, including those filed after the newest 10-K. Oldest first.
    filings are ingest.get_filings() rows, newest first, each with report_date.
    """
    start = window_start(oldest_10k_period).isoformat()
    chosen = [f for f in filings if f.get('report_date') and f['report_date'] > start]
    return sorted(chosen, key=lambda f: f['report_date'])
