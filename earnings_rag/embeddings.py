from openai import OpenAI
from earnings_rag.config import settings
import json

from earnings_rag.store import connect

_client_instance = None
def _client() -> OpenAI:
    global _client_instance
    if _client_instance is None:
        if not settings.embedding_api_key:
            raise RuntimeError('EMBEDDING_API_KEY not set')
        _client_instance = OpenAI(api_key=settings.embedding_api_key)
    return _client_instance

def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = _client().embeddings.create(
        model=settings.embedding_model,
        input=texts
    )

    vectors = []
    for item in resp.data:
        vectors.append(item.embedding)

    return vectors

def embed_batched(texts: list[str]) -> list[list[float]]:
    all_vectors = []
    batch_size = settings.embedding_batch_size

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vectors = embed_texts(batch)
        all_vectors.extend(vectors)
        print(f'embedded {start + len(batch)}/{len(texts)}')

    return all_vectors

def upsert_chunks(records: list[dict], vectors: list[list[float]]) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            for record, vector in zip(records, vectors):
                cur.execute(
                    """
                    INSERT INTO CHUNKS (id, ticker, period, chunk_index, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding
                    """,
                    (record['id'], records['ticker'], record['period'],
                     record['chunk_index'], record['text'], str(vector))
                )
        conn.commit()


if __name__ == '__main__':
    vecs = embed_texts(['hello world', 'the cat sat on the mat'])
    print(len(vecs), len(vecs[0]))



