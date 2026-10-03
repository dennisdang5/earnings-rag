"""
Relative periods ("next fiscal year", "last quarter", "two years ago") resolved in code from today's date and each
company's fiscal calendar. The model got these wrong and ignored corrections in the system prompt (next fiscal year
read as FY2027, 5 of 5 runs, even beside a table saying FY2028), but follows a note beside the question (FY2028, 5 of 5),
so run_agent appends period_note() to the user message. Pure: the calendar comes from text_periods() rows.
"""
import re
from datetime import date

from earnings_rag.agent.citations import years_in
from earnings_rag.pipeline import COMPANY_PATTERNS

NUMBERS = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
OFFSETS = {'this': 0, 'current': 0, 'next': 1, 'last': -1, 'previous': -1, 'prior': -1}

# Order matters: the annual-report phrase first, so "this year's annual report" is not also read as "this year".
PHRASE = re.compile(
    r"(?P<annual>this year'?s (?:annual report|10-?k)|(?:annual report|10-?k) (?:for|from) this year)"
    r"|(?P<ago>(?P<n>an?|one|two|three|four|five|\d+) (?:fiscal )?(?P<unit>years?|quarters?) (?P<dir>ago|from now))"
    r"|(?P<rel>(?P<which>this|current|next|last|previous|prior) (?:fiscal )?(?P<what>year|quarter))\b",
    re.IGNORECASE)


def add_months(d: date, n: int) -> date:
    """d moved n months, the day clamped to the month's length (Jan 31 + 1 month = Feb 28)."""
    y, m = divmod(d.month - 1 + n, 12)
    year, month = d.year + y, m + 1
    days = [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31][month - 1]
    return date(year, month, min(d.day, days))


class Calendar:
    """
    One company's fiscal calendar, anchored on its newest 10-K: fiscal year y ends 12 months after y - 1, quarters
    every 3 months. 52/53-week years (NVIDIA, Apple) drift a few days from this, which can mislabel a date only in the
    few days around a quarter end.
    """
    def __init__(self, newest_10k: dict):
        self.anchor_year = newest_10k['fiscal_year']
        self.anchor_end = date.fromisoformat(str(newest_10k['period']))

    def year_end(self, fy: int) -> date:
        return add_months(self.anchor_end, 12 * (fy - self.anchor_year))

    def quarter_end(self, fy: int, q: int) -> date:
        return add_months(self.year_end(fy - 1), 3 * q)

    def containing(self, d: date) -> tuple[int, int]:
        """(fiscal year, quarter) that date d falls in."""
        fy = self.anchor_year + (d.year - self.anchor_end.year) - 1
        while self.year_end(fy) < d:
            fy += 1
        return fy, next(q for q in (1, 2, 3, 4) if self.quarter_end(fy, q) >= d)


def calendars(filings: list[dict]) -> dict[str, tuple[Calendar, dict]]:
    """Each company's calendar and newest 10-K row, from text_periods() rows (oldest first)."""
    newest = {r['ticker']: r for r in filings if r['form'] == '10-K'}
    return {t: (Calendar(r), r) for t, r in newest.items()}


def shift_quarter(fy: int, q: int, n: int) -> tuple[int, int]:
    i = fy * 4 + q - 1 + n
    return i // 4, i % 4 + 1


def signed(m: re.Match) -> int:
    """The count in "two years ago" / "three quarters from now", negative for ago."""
    n = NUMBERS.get(m['n'].lower()) or int(m['n'])
    return n if m['dir'].lower() == 'from now' else -n


def resolve_periods(question: str, filings: list[dict], today: date) -> list[dict]:
    """
    Every relative period in the question, resolved for each company it names (all companies if it names none):
    [{'phrase', 'ticker', 'label', 'fiscal_year', 'period'}]. Nothing is resolved when the question also gives an
    explicit year: "fiscal 2025 compared with the prior year" is relative to 2025, not to today.
    """
    if years_in(question):
        return []
    cals = calendars(filings)
    named = [t for t, p in COMPANY_PATTERNS.items() if p.search(question) and t in cals] or list(cals)
    out = []
    for m in PHRASE.finditer(question):
        for ticker in named:
            cal, newest = cals[ticker]
            fy, q = cal.containing(today)
            extra = ''
            if m['annual']:                            # user's decision: "this year's annual report" = the newest 10-K
                fy, period = newest['fiscal_year'], 'FY'
            elif m['ago'] and m['unit'].lower().startswith('quarter'):
                fy, q = shift_quarter(fy, q, signed(m))
                period = f'Q{q}'
            elif m['ago']:                             # the fiscal year that today's date, n years away, falls in;
                then = add_months(today, 12 * signed(m))   # its quarter too, for a 10-Q question
                fy, q = cal.containing(then)
                period, extra = 'FY', f' ({then.isoformat()} falls in its Q{q})'
            elif m['what'].lower() == 'quarter':       # user's decision: "last quarter" = the last completed one
                fy, q = shift_quarter(fy, q, OFFSETS[m['which'].lower()])
                period = f'Q{q}'
            else:
                fy, period = fy + OFFSETS[m['which'].lower()], 'FY'
            label = (f'{ticker} FY{fy}' if period == 'FY' else f'{ticker} {period} FY{fy}') + extra
            out.append({'phrase': m.group(0), 'ticker': ticker, 'label': label, 'fiscal_year': fy, 'period': period})
    return out


def period_note(resolved: list[dict], today: date) -> str:
    """'(Resolved from today's date, 2026-10-02: "next fiscal year" means NVDA FY2028.)' or '' if nothing resolved."""
    if not resolved:
        return ''
    by_phrase: dict[str, list[str]] = {}
    for r in resolved:
        by_phrase.setdefault(r['phrase'], []).append(r['label'])
    parts = [f'"{p}" means {", ".join(labels)}' for p, labels in by_phrase.items()]
    return f"(Resolved from today's date, {today.isoformat()}: {'; '.join(parts)}.)"
