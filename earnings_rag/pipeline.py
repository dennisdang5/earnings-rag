import json
from earnings_rag.config import settings
from earnings_rag.embeddings import embed_batched
from earnings_rag.store import init_schema, upsert_chunks

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

if __name__ == '__main__':
    init_schema()
    build_index()