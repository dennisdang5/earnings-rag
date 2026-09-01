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
    params = []
    for record, vector in zip(records, vectors):
        params.append((record['id'], record['ticker'], record['period'],
                       record['chunk_index'], record['text'], str(vector)))

    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO CHUNKS (id, ticker, period, chunk_index, text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding
                """,
                params,
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

def sample_chunks(n: int = 1, ticker: str | None = None) -> None:
    """Print n random chunks from the corpus that can optionally be filtered by ticker

    A read and inspect helper for building out eval harness set. Sample a chunk, read it, then write a question it answers and record
    its id as ground truth.

    Prints to stdout rather than returning
    """
    sql = 'SELECT id, text FROM chunks'
    params = []
    if ticker:
        sql += ' WHERE ticker = %s'
        params.append(ticker)
    sql += ' ORDER BY random() LIMIT %s'
    params.append(n)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            for row in cur.fetchall():
                print(f'=== {row[0]} ===')
                print(row[1][:1200])
                print()

def show_chunk(chunk_id: str) -> None:
    """ Print one chunk's full text by id and is used for eval work."""
    chunk = get_chunk(chunk_id)
    if chunk is None:
        print(f'No chunk with id {chunk_id}')
        return
    print(f'==={chunk['id']}===')
    print(chunk['text'])

def get_chunk(chunk_id: str) -> dict | None:
    """
    Fetch one chunk by id and returns None if it doesn't exist.
    This returns data for API callers rather than printing in show chunk.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT id, ticker, period, chunk_index, text FROM chunks where id = %s',
                (chunk_id),
            )
            row = cur.fetchone()

    if row is None:
        return None

    return {
        'id': row[0],
        'ticker': row[1],
        'period': row[2],
        'chunk_index': row[3],
        'text': row[4]
    }

def init_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()

if __name__ == '__main__':
    # sample_chunks(4, ticker='NVDA')

    show_chunk('AAPL_2024-09-28_0014')
    print()
    show_chunk('AAPL_2022-09-24_0014')
    print()
    show_chunk('AAPL_2023-09-30_0014')
    print()
    show_chunk('AAPL_2024-09-28_0013')
    print()
    show_chunk('AAPL_2025-09-27_0014')