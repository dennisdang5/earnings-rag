from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from earnings_rag.config import settings
from earnings_rag.pipeline import ask as run_ask
from earnings_rag.store import get_chunk
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    ticker: str | None = None

class Source(BaseModel):
    id: str
    ticker: str
    period: str
    distance: float
    excerpt: str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

class Chunk(BaseModel):
    id: str
    ticker: str
    period: str
    chunk_index: int
    text: str

ASK_REQUESTS = Counter(
    'earnings_rag_ask_requests_total',
    'Total /ask requests',
    ['status']
)

ASK_LATENCY = Histogram(
    'earnings_rag_ask_duration_seconds',
    'End-to-end /ask latency',
    buckets=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
)

RETRIEVAL_DISTANCE = Histogram(
    'earnings_rag_top_distance',
    'Cosine distance of the top retrieved chunk',
    buckets=(0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)
)
app = FastAPI(title='Earnings RAG', version='0.1.0')

EXCERPT_CHARS = settings.excerpt_chars

@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}

@app.post('/ask', response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    result = run_ask(req.question, k=req.k, ticker=req.ticker)

    sources = []
    for hit in result['sources']:
        sources.append(Source(
            id=hit['id'],
            ticker=hit['ticker'],
            period=hit['period'],
            distance=hit['distance'],
            excerpt=hit['text'][:EXCERPT_CHARS]
        ))

    return AskResponse(answer=result['answer'], sources=sources)

@app.get('/chunks/{chunk_id}', response_model=Chunk)
def chunk_endpoint(chunk_id: str) -> Chunk:
    chunk = get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f'No chunk with id {chunk_id}')
    return Chunk(**chunk)

@app.get('/metrics')
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

