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

def init_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()

if __name__ == '__main__':
    init_schema()
    print('schema ready')