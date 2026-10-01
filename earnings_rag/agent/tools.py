import json
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from earnings_rag.calc import evaluate
from earnings_rag.pipeline import retrieve
from earnings_rag.store import get_facts, fact_years
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


def search_filings(args: SearchFilingsArgs) -> dict:
    # route=False: the model decides the company explicitly, so keyword routing must not second-guess it
    hits = retrieve(args.query, ticker=args.company, route=False)
    return {
        'results': [
            {'id': h['id'], 'ticker': h['ticker'], 'period': h['period'],
             'distance': round(h['distance'], 4), 'text': h['text']}
            for h in hits
        ]
    }


SEARCH_FILINGS = Tool(
    name='search_filings',
    description=(
        'Semantic search over the text of NVIDIA, Apple, and Capital One annual reports (10-K). '
        'Returns the 5 closest passages, each with an id you can cite. Tables are not included: '
        'for financial figures, use get_financials.'
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
AVAILABLE = {t: [m for m, spec in METRICS.items() if t in spec['concepts']] for t in COMPANIES}
DEFAULT_FACT_ROWS = 8  # the latest two years of quarters and their FY rows; every row is re-sent on later calls

MetricName = Literal[tuple(METRICS)]  # the model can only name metrics that exist


class GetFinancialsArgs(BaseModel):
    company: Literal[COMPANIES]
    metric: MetricName
    fiscal_year: int | None = Field(default=None, description='Omit for the most recent periods.')
    period: Literal['FY', 'Q1', 'Q2', 'Q3', 'Q4'] | None = Field(
        default=None, description='FY for the full year, Q1-Q4 for a quarter. Omit for all periods.')


def fact_result(row: dict) -> dict:
    """One facts row as the model sees it: dollars in millions, provenance, and the as-reported figure if adjusted."""
    period = row['fiscal_period']
    value, unit = row['value'], row['unit']
    if unit == 'USD':
        value, unit = value / 1e6, 'USD millions'   # the 10-Ks report in millions; long numbers invite digit errors
        value = int(value) if value.is_integer() else round(value, 3)
    elif unit == 'USD/shares':
        unit = 'USD per share'

    out = {'id': f"{row['ticker']}_{row['metric']}_FY{row['fiscal_year']}{'' if period == 'FY' else period}",
           'company': row['ticker'], 'metric': row['metric'], 'fiscal_year': row['fiscal_year'], 'period': period,
           'period_end': str(row['period_end']), 'value': value, 'unit': unit, 'derived': row['derived'],
           'source': f"{row['form']} filed {row['filed']}"}
    if row['split_factor'] != 1:
        # the filing shows the pre-split figure; giving both lets the answer match the document it cites
        out['split_adjusted'] = True
        out['as_reported'] = round(row['value'] * row['split_factor'], 6)
    return out


def get_financials(args: GetFinancialsArgs) -> dict:
    if args.metric not in AVAILABLE[args.company]:
        return {'error': f'{args.company} does not report {args.metric}. '
                         f'Available for {args.company}: {", ".join(AVAILABLE[args.company])}'}

    rows = get_facts(args.company, args.metric, args.fiscal_year, args.period, limit=DEFAULT_FACT_ROWS)
    if not rows:
        years = fact_years(args.company, args.metric)
        span = f'fiscal years {years[0]}-{years[1]}' if years else 'no years'
        return {'results': [], 'note': f'No matching data. {args.company} {args.metric} is available for {span}.'}
    return {'results': [fact_result(r) for r in rows]}


GET_FINANCIALS = Tool(
    name='get_financials',
    description=(
        'Reported financial figures from SEC XBRL data, each with an id you can cite. '
        'Fiscal year is the calendar year the fiscal year ends in (NVIDIA fiscal 2025 ended January 2025). '
        'Dollar amounts are in USD millions. Per-share values are adjusted for stock splits to today\'s share count; '
        'adjusted rows also give the as_reported figure. derived=true means a quarter computed as a year-to-date '
        'total minus the earlier one. Metrics per company: '
        + '; '.join(f'{t}: {", ".join(ms)}' for t, ms in AVAILABLE.items()) + '.'
    ),
    args_model=GetFinancialsArgs,
    fn=get_financials,
)

DEFAULT_TOOLS = [SEARCH_FILINGS, GET_FINANCIALS, CALCULATE]
