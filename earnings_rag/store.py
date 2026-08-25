import psycopg
from earnings_rag.config import settings

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks(
    id  TEXT    PRIMARY KEY,
    ticker  TEXT    NOT NULL,
    period  TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    text    TEXT    NOT NULL,
    embedding   vector({settings.embedding_dim})
    );

CREATE INDEX IF NOT EXISTS chunks_ticker_period_idx ON chunks (ticker, period);
"""

def connect() -> psycopg.Connection:
    return psycopg.connect(settings.db_url)

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
                    (record['id'], record['ticker'], record['period'],
                     record['chunk_index'], record['text'], str(vector))
                )
        conn.commit()

def search(query_vector: list[float], k: int = 5, ticker: str | None = None) -> list[dict]:
    vector = str(query_vector)

    if ticker:
        sql = """
            SELECT id, ticker, period, text, embedding <=> %s::vector AS distance
            FROM chunks
            WHERE ticker = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = (vector, ticker, vector, k)
    else:
        sql = """
            SELECT id, ticker, period, text, embedding <=> %s::vector AS distance
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = (vector, vector, k)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    results = []
    for row in rows:
        results.append({
            'id': row[0],
            'ticker': row[1],
            'period': row[2],
            'text': row[3],
            'distance': row[4]
        })

    return results

def init_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()

if __name__ == '__main__':
    init_schema()
    print('schema ready')