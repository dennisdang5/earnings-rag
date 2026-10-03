import json
import re
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from earnings_rag.calc import evaluate
from earnings_rag.pipeline import retrieve
from earnings_rag.segments import CORPORATE, find_parts
from earnings_rag.store import (get_facts, fact_years, get_breakdown, available_breakdowns, breakdown_names, available_metrics,
                                text_periods)
from earnings_rag.xbrl import METRICS


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]  # one class both describes the tool to the model and validates its arguments
    fn: Callable[[BaseModel], dict]

    def schema(self) -> dict:
        """The tool definition in the shape the OpenAI API expects."""
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.args_model.model_json_schema(),
            },
        }

    def call(self, raw_arguments: str) -> str:
        """
        Run the tool on the model's raw JSON argument string and return a JSON string.

        Never raises: errors are returned as {"error": ...} so the model can read them and retry,
        instead of one bad call killing the whole request.
        """
        try:
            args = self.args_model.model_validate_json(raw_arguments)
        except ValidationError as e:
            return json.dumps({'error': f'Invalid arguments: {e.errors(include_url=False)}'}, default=str)

        try:
            return json.dumps(self.fn(args))
        except Exception as e:
            return json.dumps({'error': f'{self.name} failed: {e}'})


class SearchFilingsArgs(BaseModel):
    query: str = Field(description='What to look for, phrased as a topic or question.')
    company: Literal['NVDA', 'AAPL', 'COF'] | None = Field(
        default=None,
        description='Ticker to restrict the search to. Omit to search all companies.',
    )
    fiscal_year: int | None = Field(
        default=None, description='Fiscal year (the calendar year it ends in). Omit for all years.')
    period: Literal['FY', 'Q1', 'Q2', 'Q3', 'Q4'] | None = Field(
        default=None, description=('Q1, Q2 or Q3 searches that quarter\'s 10-Q (quarterly report). FY, or Q4, searches the '
                                   'annual report (10-K), which also covers the fourth quarter. Needs fiscal_year. Omit to '
                                   'search annual reports only.'))
    latest: bool = Field(
        default=False, description=('true searches only each company\'s most recent filing, quarterly or annual. Use it for '
                                    '"latest", "most recent" or "current" questions. Not combinable with fiscal_year or '
                                    'period.'))


def describe_periods(rows: list[dict]) -> str:
    """'NVDA: 10-K FY2023-FY2026 (4); 10-Q Q1 FY2023-Q2 FY2027 (14)' from text_periods() rows."""
    by_company: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_company.setdefault(r['ticker'], {}).setdefault(r['form'], []).append(r)
    parts = []
    for ticker, forms in by_company.items():
        spans = []
        for form, filings in forms.items():
            first, last = filings[0], filings[-1]
            label = (lambda r: f"FY{r['fiscal_year']}" if r['fiscal_period'] == 'FY'
                     else f"{r['fiscal_period']} FY{r['fiscal_year']}")
            spans.append(f'{form} {label(first)}-{label(last)} ({len(filings)})')
        parts.append(f'{ticker}: ' + '; '.join(spans))
    return ' | '.join(parts)


def missing_label(company: str | None, fiscal_year: int | None, period: str | None, form: str | None) -> str:
    """'NVDA Q2 FY2030 10-Q' for the filing a search asked for and did not find. The loop checks the answer admits it."""
    if form is None:                                  # latest=true: whichever filing is newest
        when = 'latest'
    elif fiscal_year is None:
        when = period or ''
    elif period in ('Q1', 'Q2', 'Q3'):
        when = f'{period} FY{fiscal_year}'
    else:
        when = f'FY{fiscal_year}'
    return ' '.join(p for p in (company, when, form or 'filing') if p)


def search_filings(args: SearchFilingsArgs) -> dict:
    if args.latest and (args.fiscal_year is not None or args.period is not None):
        return {'error': 'latest cannot be combined with fiscal_year or period: use one or the other.'}
    if args.period is not None and args.fiscal_year is None:
        # A quarter alone searched it across every year: "last quarter" at NVIDIA cited Q2 FY2027 from passages of which
        # 2 of 5 were Q2 FY2026.
        return {'error': 'period needs fiscal_year: a quarter alone would mix the filings of every year for that quarter. Set '
                         'fiscal_year, or use latest=true for the newest filing.'}

    notes = []
    period = args.period
    if period == 'Q4':
        period = 'FY'
        notes.append('Q4 has no quarterly report of its own: its text is in the annual report (10-K), which was searched.')
    # a quarter means 10-Qs; otherwise annual reports, the default; "latest" looks at whichever form is newest
    form = '10-Q' if period in ('Q1', 'Q2', 'Q3') else (None if args.latest else '10-K')
    if args.period is None and not args.latest:
        notes.append('Searched annual reports (10-K) only. For a quarter, or the latest or most recent report, set period '
                     '(Q1-Q3) or latest=true.')

    # route=False: the model decides the company explicitly, so keyword routing must not second-guess it
    hits = retrieve(args.query, ticker=args.company, route=False, form=form, fiscal_year=args.fiscal_year,
                    fiscal_period=period, latest=args.latest)
    out = {'results': [
        {'id': h['id'], 'ticker': h['ticker'], 'form': h.get('form'), 'period': h['period'],
         'fiscal_year': h.get('fiscal_year'), 'fiscal_period': h.get('fiscal_period'),
         'distance': round(h['distance'], 4), 'text': h['text']}
        for h in hits
    ]}
    if not hits:
        # An error, not an empty result: with a note the model answered from the latest filing when asked for Q2 FY2030
        # (3 of 3 runs). The filing asked for is not in the corpus, which is a fact to report, not a quiet miss.
        return {'error': 'No filing matches those filters, so the filing asked for is not available. Tell the user that '
                         'first; if you then answer from a different period, say which one. Filings with text: '
                         + describe_periods(text_periods(args.company)),
                'missing_filing': {'label': missing_label(args.company, args.fiscal_year, period, form),
                                   'fiscal_year': args.fiscal_year}}
    if notes:
        out['note'] = ' '.join(notes)
    return out


SEARCH_FILINGS = Tool(
    name='search_filings',
    description=(
        'Semantic search over the text of NVIDIA, Apple, and Capital One annual reports (10-K) and quarterly reports '
        '(10-Q). Annual reports are searched unless you set period (Q1-Q3, the quarterly reports) or latest=true '
        '(the most recent filing). Returns the 5 closest passages, each with an id you can cite and the filing it comes '
        'from (form, fiscal year and period). Tables are not included: for financial figures, use get_financials.'
    ),
    args_model=SearchFilingsArgs,
    fn=search_filings,
)

class CalculateArgs(BaseModel):
    expression: str = Field(description=(
        'Arithmetic using numbers, + - * / ** and parentheses only. No %, $, units or thousands separators. '
        'Example, percent growth: (60922 - 26974) / 26974 * 100'
    ))


def calculate(args: CalculateArgs) -> dict:
    return {'expression': args.expression, 'result': evaluate(args.expression)}


CALCULATE = Tool(
    name='calculate',
    description=(
        'Exact arithmetic. Use it for every derived number (growth, margins, differences, ratios) '
        'instead of computing in your head.'
    ),
    args_model=CalculateArgs,
    fn=calculate,
)

COMPANIES = ('NVDA', 'AAPL', 'COF')
DEFAULT_FACT_ROWS = 8  # the latest two years of quarters and their FY rows; every row is re-sent on later calls
BREAKDOWN_YEARS = 2    # without a fiscal year, a breakdown covers two years so a growth question takes one call

MetricName = Literal[tuple(METRICS)]  # the model can only name metrics that exist


class GetFinancialsArgs(BaseModel):
    company: Literal[COMPANIES]
    metric: MetricName
    fiscal_year: int | None = Field(default=None, description='Omit for the most recent periods.')
    period: Literal['FY', 'Q1', 'Q2', 'Q3', 'Q4'] | None = Field(
        default=None, description='FY for the full year, Q1-Q4 for a quarter. Omit for all periods.')
    breakdown: Literal['product', 'segment', 'geography'] | None = Field(
        default=None, description=(
            'Split the figure into slices. product: product lines or markets (e.g. Data Center, Gaming, iPhone, '
            'Services). segment: the reporting segments, which are regions for some companies (e.g. Compute and '
            "Networking; Apple's Americas, Europe, Greater China; Credit Card). geography: revenue by country, "
            'usually a few named countries plus an "Other Countries" remainder that is not a region. Returns every '
            'slice for the year (the latest two years if fiscal_year is omitted). Annual only.'))


def fact_result(row: dict) -> dict:
    """One facts row as the model sees it: dollars in millions, provenance, and the as-reported figure if adjusted."""
    period = row['fiscal_period']
    value, unit = row['value'], row['unit']
    if unit == 'USD':
        value, unit = value / 1e6, 'USD millions'   # the 10-Ks report in millions; long numbers invite digit errors
        value = int(value) if value.is_integer() else round(value, 3)
    elif unit == 'USD/shares':
        unit = 'USD per share'

    fact_id = f"{row['ticker']}_{row['metric']}_FY{row['fiscal_year']}{'' if period == 'FY' else period}"
    out = {'id': fact_id, 'company': row['ticker'], 'metric': row['metric'], 'fiscal_year': row['fiscal_year'],
           'period': period}
    if row.get('axis'):
        slug = ''.join(w[:1].upper() + w[1:] for w in re.split(r'[^A-Za-z0-9]+', row['segment']) if w)
        out['id'] = f"{fact_id}_{row['axis']}_{slug}"   # NVDA_revenue_FY2025_product_DataCenter
        out['breakdown'], out['segment'] = row['axis'], row['segment']
    if METRICS.get(row['metric'], {}).get('balance'):
        out['balance'] = True                            # a snapshot on one date: never add it across years or periods
        out['as_of'] = str(row['period_end'])
    out |= {'period_end': str(row['period_end']), 'value': value, 'unit': unit, 'derived': row['derived'],
            'source': f"{row['form']} filed {row['filed']}"}
    if row['split_factor'] != 1:
        # the filing shows the pre-split figure; giving both lets the answer match the document it cites
        out['split_adjusted'] = True
        out['as_reported'] = round(row['value'] * row['split_factor'], 6)
    return out


def get_breakdown_result(args: GetFinancialsArgs) -> dict:
    if args.period not in (None, 'FY'):
        return {'error': 'Breakdowns are annual only for now: omit period or use FY.'}
    available = available_breakdowns(args.company, CORPORATE)
    if (args.metric, args.breakdown) not in {(m, a) for m, a, _, _ in available}:
        listed = '; '.join(f'{m} by {a} (FY{lo}-FY{hi})' for m, a, lo, hi in available) or 'none'
        return {'error': f'{args.company} does not report {args.metric} by {args.breakdown}. '
                         f'Breakdowns available for {args.company}: {listed}'}

    rows = get_breakdown(args.company, args.metric, args.breakdown, args.fiscal_year, BREAKDOWN_YEARS)
    if not rows:
        lo, hi = next((lo, hi) for m, a, lo, hi in available if (m, a) == (args.metric, args.breakdown))
        return {'results': [], 'note': f'No matching data. This breakdown is available for fiscal years {lo}-{hi}.'}
    parts = find_parts(rows)
    results = []
    for r in rows:
        out = fact_result(r)
        if (r['fiscal_year'], r['segment']) in parts:
            out['part_of'] = parts[(r['fiscal_year'], r['segment'])]
        results.append(out)
    notes = []
    if parts:
        notes.append('Rows with part_of are already included in that row: do not add them to it.')
    # Without this the model asked for "segment", got Compute and Networking, and called it Data Center (3 of 3 runs)
    other = {a: names for a, names in breakdown_names(args.company, args.metric).items()
             if a != args.breakdown and len([n for n in names if n != CORPORATE]) >= 2}
    if other:
        notes.append(f'Other breakdowns of {args.company} {args.metric}: '
                     + '; '.join(f'{a}: {", ".join(names)}' for a, names in other.items())
                     + '. If the slice you need is listed there, call again with that breakdown.')
    return {'results': results, **({'note': ' '.join(notes)} if notes else {})}


def get_financials(args: GetFinancialsArgs) -> dict:
    if args.breakdown is not None:
        return get_breakdown_result(args)
    if METRICS[args.metric].get('balance') and args.period == 'Q4':
        return {'error': f'{args.metric} is a balance on a date: the fiscal year end is period FY, '
                         f'and Q1-Q3 are the quarter ends.'}
    available = available_metrics(args.company)
    if args.metric not in available:
        return {'error': f'{args.company} does not report {args.metric}. '
                         f'Available for {args.company}: {", ".join(available)}'}

    rows = get_facts(args.company, args.metric, args.fiscal_year, args.period, limit=DEFAULT_FACT_ROWS)
    if not rows:
        years = fact_years(args.company, args.metric)
        span = f'fiscal years {years[0]}-{years[1]}' if years else 'no years'
        return {'results': [], 'note': f'No matching data. {args.company} {args.metric} is available for {span}.'}
    out = {'results': [fact_result(r) for r in rows]}
    if METRICS[args.metric].get('profit') and not args.breakdown:
        # The model fetched gross profit, then cost of revenue, and divided by the cost: -14.10% (4 of 8 runs)
        out['note'] = f'A {args.metric.replace("_", " ")} margin is this figure divided by revenue.'
    # Without this the model computed Apple's company-wide margin and called it Services', never learning that cost of
    # revenue can be split by product (3 of 3 runs)
    available = available_breakdowns(args.company, CORPORATE)
    if available:
        by_metric: dict[str, list[str]] = {}
        for metric, axis, _, _ in available:
            by_metric.setdefault(metric, []).append(axis)
        out['note'] = (out.get('note', '') + f' Breakdowns available for {args.company} (set breakdown to get one slice such as a product '
                       f'line or segment): ' + '; '.join(f'{m} by {", ".join(axes)}' for m, axes in by_metric.items()))
    return out


GET_FINANCIALS = Tool(
    name='get_financials',
    description=(
        'Reported financial figures from SEC XBRL data, each with an id you can cite. '
        'Fiscal year is the calendar year the fiscal year ends in (NVIDIA fiscal 2025 ended January 2025). '
        'Dollar amounts are in USD millions. Per-share values are adjusted for stock splits to today\'s share count; '
        'adjusted rows also give the as_reported figure. derived=true means a quarter computed as a year-to-date '
        'total minus the earlier one. balance=true rows (loans, deposits) are values on the as_of date, not amounts '
        'over a period: compare them across dates, never add them. If the question is about one product line, segment or '
        'region (Services, Data Center, Americas, Credit Card), set breakdown: a company-wide figure is not a '
        'substitute for part of the company, and a margin for that part needs its revenue and its cost of revenue, '
        'both with the same breakdown. '
        'A margin is a profit metric divided by revenue (gross profit / revenue), never by a cost. '
        'Metrics: ' + ', '.join(METRICS) + '. Not every company reports every metric (a bank has no gross profit); '
        'asking for one it does not report returns an error listing what it does.'
    ),
    args_model=GetFinancialsArgs,
    fn=get_financials,
)

DEFAULT_TOOLS = [SEARCH_FILINGS, GET_FINANCIALS, CALCULATE]
