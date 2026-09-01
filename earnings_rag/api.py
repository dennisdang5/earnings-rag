from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from earnings_rag.config import settings
from earnings_rag.pipeline import ask as run_ask
from earnings_rag.store import get_chunk

app = FastAPI(title='Earnings RAG', version='0.1.0')

EXCERPT_CHARS = settings.excerpt_chars

@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}

# @app.post('/ask', response_model=AskResponse)
# def ask_endpoint(req: AskRequest) -> AskResponse):
