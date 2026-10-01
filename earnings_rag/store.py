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

-- segment '' means consolidated; per-segment rows (e.g. a product line) share the table
CREATE TABLE IF NOT EXISTS facts(
    ticker  TEXT    NOT NULL,
    metric  TEXT    NOT NULL,
    segment TEXT    NOT NULL DEFAULT '',
    fiscal_year INTEGER NOT NULL,
    fiscal_period   TEXT    NOT NULL,
    concept TEXT    NOT NULL,
    unit    TEXT    NOT NULL,
    value   DOUBLE PRECISION NOT NULL,
    period_start    DATE    NOT NULL,
    period_end  DATE    NOT NULL,
    derived BOOLEAN NOT NULL,
    form    TEXT    NOT NULL,
    accession   TEXT    NOT NULL,
    filed   DATE    NOT NULL,
    split_factor    DOUBLE PRECISION NOT NULL DEFAULT 1,
    PRIMARY KEY (ticker, metric, segment, fiscal_year, fiscal_period)
    );

-- value * split_factor is the as-reported figure. This line migrates databases created before the column existed.
ALTER TABLE facts ADD COLUMN IF NOT EXISTS split_factor DOUBLE PRECISION NOT NULL DEFAULT 1;
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
    print(f'==={chunk["id"]}===')
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
                (chunk_id,),
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

FACT_COLUMNS = ['ticker', 'metric', 'segment', 'fiscal_year', 'fiscal_period', 'concept', 'unit', 'value',
                'period_start', 'period_end', 'derived', 'form', 'accession', 'filed', 'split_factor']

def upsert_facts(rows: list[dict]) -> None:
    sql = f"""
        INSERT INTO facts ({', '.join(FACT_COLUMNS)})
        VALUES ({', '.join(['%s'] * len(FACT_COLUMNS))})
        ON CONFLICT (ticker, metric, segment, fiscal_year, fiscal_period) DO UPDATE SET
            {', '.join(f'{c} = EXCLUDED.{c}' for c in FACT_COLUMNS[5:])}
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, [tuple(r[c] for c in FACT_COLUMNS) for r in rows])
        conn.commit()

def get_facts(ticker: str, metric: str, fiscal_year: int | None = None, fiscal_period: str | None = None,
              segment: str = '', limit: int = 12) -> list[dict]:
    """
    Most recent first. With no fiscal_year, this returns the latest periods, which answers "last quarter" questions.
    """
    sql = f'SELECT {", ".join(FACT_COLUMNS)} FROM facts WHERE ticker = %s AND metric = %s AND segment = %s'
    params = [ticker, metric, segment]
    if fiscal_year is not None:
        sql += ' AND fiscal_year = %s'
        params.append(fiscal_year)
    if fiscal_period is not None:
        sql += ' AND fiscal_period = %s'
        params.append(fiscal_period)
    sql += ' ORDER BY period_end DESC, fiscal_period LIMIT %s'  # FY sorts before Q1.. on the same end date
    params.append(limit)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(zip(FACT_COLUMNS, row)) for row in cur.fetchall()]

def init_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()
