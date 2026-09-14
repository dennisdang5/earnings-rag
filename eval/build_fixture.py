import json
import random
from earnings_rag.config import REPO_ROOT
from earnings_rag.store import connect
from earnings_rag.questions import load_questions

FIXTURE_PATH = REPO_ROOT / 'eval' / 'fixture_chunks.jsonl'

# Chunks that compete with ground truth in the three known misses
COMPETITORS = [
    "COF_2022-12-31_0217", "COF_2023-12-31_0235", "COF_2024-12-31_0244",
    "COF_2024-12-31_0245", "COF_2025-12-31_0266",
    "COF_2022-12-31_0125", "COF_2024-12-31_0148", "COF_2024-12-31_0201",
    "COF_2024-12-31_0202", "COF_2025-12-31_0224",
    "COF_2023-12-31_0050", "COF_2024-12-31_0054", "COF_2024-12-31_0055",
    "COF_2024-12-31_0206", "COF_2025-12-31_0057",
]

TARGET_TOTAL = 200

def build_fixture() -> None:
    questions = load_questions(REPO_ROOT / 'eval' / 'questions.yaml')

    wanted = set(COMPETITORS)
    for q in questions:
        for chunk_id in (q.get('expected_chunks') or []):
            wanted.add(chunk_id)

    print(f'{len(wanted)} required chunks')

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT id FROM chunks WHERE id != ALL(%s) ORDER BY random() LIMIT %s',
                (list(wanted), TARGET_TOTAL - len(wanted)),
            )
            for row in cur.fetchall():
                wanted.add(row[0])

            cur.execute(
                """SELECT id, ticker, period, chunk_index, text, embedding::text
                FROM chunks WHERE id = ANY(%s) ORDER BY id""",
                (list(wanted),),
            )
            rows = cur.fetchall()

    with FIXTURE_PATH.open('w', encoding='utf-8') as f:
        for row in rows:
            record = {
                'id': row[0],
                'ticker': row[1],
                'period': row[2],
                'chunk_index': row[3],
                'text': row[4],
                'embedding': json.loads(row[5])
            }
            f.write(json.dumps(record) + '\n')

        print(f'Wrote {len(rows)} chunks to {FIXTURE_PATH}')


if __name__ == '__main__':
    build_fixture()