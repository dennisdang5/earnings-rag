import json
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from earnings_rag.pipeline import retrieve


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
        'Returns the 5 closest passages, each with an id you can cite. Tables are not included, '
        'so this cannot answer questions about specific numbers.'
    ),
    args_model=SearchFilingsArgs,
    fn=search_filings,
)

DEFAULT_TOOLS = [SEARCH_FILINGS]
