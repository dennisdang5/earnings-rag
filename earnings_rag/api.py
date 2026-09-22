from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from earnings_rag.config import settings
from earnings_rag.pipeline import ask as run_ask, retrieve
from earnings_rag.store import get_chunk
from earnings_rag.llm import generate_stream
import time
import json
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog

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

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.processors.JSONRenderer()
    ]
)
log = structlog.get_logger()

def sse(event: str, data) -> str:
    return f'event: {event}\ndata: {json.dumps(data)}\n\n'

app = FastAPI(title='Earnings RAG', version='0.1.0')

EXCERPT_CHARS = settings.excerpt_chars

@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}

@app.post('/ask', response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    started = time.perf_counter()
    status = 'ok'
    sources = []

    try:
        result = run_ask(req.question, k=req.k, ticker=req.ticker)

        for hit in result['sources']:
            sources.append(Source(
                id=hit['id'],
                ticker=hit['ticker'],
                period=hit['period'],
                distance=hit['distance'],
                excerpt=hit['text'][:EXCERPT_CHARS]
            ))

        if sources:
            RETRIEVAL_DISTANCE.observe(sources[0].distance)

        return AskResponse(answer=result['answer'], sources=sources)

    except Exception:
        status = 'error'
        raise

    finally:
        elapsed = time.perf_counter() - started
        ASK_LATENCY.observe(elapsed)
        ASK_REQUESTS.labels(status=status).inc()

        log.info(
            'ask',
            question=req.question,
            k=req.k,
            ticker=req.ticker,
            status=status,
            duration_s=round(elapsed, 3),
            n_sources=len(sources) if status == 'ok' else 0,
            top_distance=round(sources[0].distance, 4) if sources else None
        )


@app.get('/chunks/{chunk_id}', response_model=Chunk)
def chunk_endpoint(chunk_id: str) -> Chunk:
    chunk = get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f'No chunk with id {chunk_id}')
    return Chunk(**chunk)

@app.get('/metrics')
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post('/ask/stream')
def ask_stream(req: AskRequest) -> StreamingResponse:
    def events():
        started = time.perf_counter()
        status = "ok"
        hits = []
        try:
            yield sse("status", {"stage": "searching"})
            hits = retrieve(req.question, k=req.k, ticker=req.ticker)
            if hits:
                RETRIEVAL_DISTANCE.observe(hits[0]["distance"])

            yield sse("sources", [
                {"id": h["id"],
                 "ticker": h["ticker"],
                 "period": h["period"],
                 "distance": h["distance"],
                 "excerpt": h["text"][:EXCERPT_CHARS]}
                for h in hits
            ])

            yield sse("status", {"stage": "writing"})
            for piece in generate_stream(req.question, hits):
                yield sse("token", {"text": piece})

            yield sse("done", {})
        except Exception:
            status = "error"
            log.exception("ask_stream_failed", question=req.question)
            yield sse("error", {"message": "Couldn't finish the answer. Try again in a moment."})
        finally:
            elapsed = time.perf_counter() - started
            ASK_LATENCY.observe(elapsed)
            ASK_REQUESTS.labels(status=status).inc()
            log.info("ask_stream", question=req.question, status=status,
                     duration_s=round(elapsed, 3), n_sources=len(hits),
                     top_distance=round(hits[0]["distance"], 4) if hits else None)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
