import json
import re
from earnings_rag.config import settings
from earnings_rag.embeddings import embed_batched, embed_texts
from earnings_rag.store import init_schema, upsert_chunks, search
from earnings_rag.llm import generate

COMPANY_PATTERNS = {
    "NVDA": re.compile(r"\b(nvidia|nvda)\b", re.IGNORECASE),
    "AAPL": re.compile(r"\b(apple|aapl)\b", re.IGNORECASE),
    "COF": re.compile(r"\b(capital one|cof)\b", re.IGNORECASE),
}

def detect_ticker(question: str) -> str | None:
    """
    Return a ticker if the question names exactly one company
    """
    found = []
    for ticker, pattern in COMPANY_PATTERNS.items():
        if pattern.search(question):
            found.append(ticker)

    if len(found) == 1:
        return found[0]

    return None

def build_index(limit: int | None = None) -> None:
    path = settings.chunks_path

    records = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    if limit:
        records = records[:limit]

    texts = []
    for r in records:
        texts.append(r['text'])

    vectors = embed_batched(texts)
    upsert_chunks(records, vectors)
    print(f'indexed {len(records)} chunks')

def retrieve(question: str, k: int = 5, ticker: str | None = None, query_vector: list[float] | None = None, route: bool = True,) -> list[dict]:
    """
    Find the k nearest chunks such that an explicit ticker wins. Otherwise, a single named company in the question limits the search
    to that company.

    route: if the question names exactly one company, search only that company's chunks. An explicit ticker always wins.
    if ticker is None and route:
        ticker = detect_ticker(question)
    """
    if ticker is None and route:
        ticker = detect_ticker(question)

    if query_vector is None:
        query_vector = embed_texts([question])[0]

    return search(query_vector, k=k, ticker=ticker)

def print_hits(hits: list[dict]) -> None:
    for hit in hits:
        print(f'--- {hit["id"]} distance={hit["distance"]:.4f} ---')
        print(hit['text'][:400])
        print()

def ask(question: str, k: int = 5, ticker: str | None = None, route: bool = True) -> dict:
    hits = retrieve(question, k=k, ticker=ticker, route=route)
    answer = generate(question, hits)
    return {'answer': answer, 'sources': hits}

if __name__ == '__main__':
    init_schema()
    build_index()