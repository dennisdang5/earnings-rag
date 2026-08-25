import json
from earnings_rag.config import settings
from earnings_rag.embeddings import embed_batched, embed_texts
from earnings_rag.store import init_schema, upsert_chunks, search


def build_index(limit: int | None = None) -> None:
    path = settings.data_dir / 'chunks.jsonl'

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

def retrieve(question: str, k: int = 5, ticker: str | None = None) -> list[dict]:
    query_vector = embed_texts([question])[0]
    return search(query_vector, k=k, ticker=ticker)

def print_hits(hits: list[dict]) -> None:
    for hit in hits:
        print(f'--- {hit['id']} distance={hit['distance']:.4f} ---')
        print(hit['text'][:400])
        print()

if __name__ == '__main__':
    # init_schema()
    # build_index()
    print_hits(retrieve('What does NVIDIA say about supply constraints?'))