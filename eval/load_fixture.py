import json
from earnings_rag.config import REPO_ROOT
from earnings_rag.store import connect, init_schema
from earnings_rag.config import settings


FIXTURE_PATH = REPO_ROOT / 'eval' / 'fixture_chunks.jsonl'

def load_fixture() -> int:
    """
    Load the CI fixture corpus into Postgres
    Returns rows loaded
    """
    if not settings.db_name.endswith("_ci"):
        raise SystemExit(
            f"Refusing to load the fixture into '{settings.db_name}'. "
            "It truncates the chunks table. Point DB_NAME at a *_ci database."
        )

    params = []
    with FIXTURE_PATH.open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            params.append((
                r['id'],
                r['ticker'],
                r['period'],
                r['chunk_index'],
                r['text'],
                str(r['embedding'])
            ))

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute('TRUNCATE chunks')
            cur.executemany(
                """
                INSERT INTO chunks (id, ticker, period, chunk_index, text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                params
            )
        conn.commit()

    return len(params)

if __name__ == '__main__':
    init_schema()
    n = load_fixture()
    print(f'Loaded {n} fixture chunks')