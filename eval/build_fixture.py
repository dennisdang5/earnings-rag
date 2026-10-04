import json
import random
from earnings_rag.config import REPO_ROOT
from earnings_rag.store import connect
from earnings_rag.questions import load_questions
from earnings_rag.embeddings import embed_texts

FIXTURE_PATH = REPO_ROOT / 'eval' / 'fixture_chunks.jsonl'
QUERY_PATH = REPO_ROOT / 'eval' / 'fixture_queries.json'

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

# Whole filings added for the quarterly questions: (ticker, form, fiscal_year, fiscal_period). `latest` in CI means the
# newest filing in the fixture, so each company's newest filing must be here in full.
QUARTERLY_FILINGS = [
    ('NVDA', '10-Q', 2027, 'Q2'),
    ('AAPL', '10-Q', 2026, 'Q3'),
    ('AAPL', '10-Q', 2025, 'Q3'),
    ('COF', '10-Q', 2026, 'Q2'),
]

def build_fixture() -> None:
    questions = load_questions(REPO_ROOT / 'eval' / 'questions.yaml')

    wanted = set(COMPETITORS)
    for q in questions:
        for chunk_id in (q.get('expected_chunks') or []):
            wanted.add(chunk_id)

    print(f'{len(wanted)} required chunks')

    with connect() as conn:
        with conn.cursor() as cur:
            if FIXTURE_PATH.exists():
                # Keep the existing sample, so the 10-K recall (and its CI gate) cannot move when the fixture grows.
                # Quarterly chunks from the sample are not kept: whole filings are added below.
                with FIXTURE_PATH.open(encoding='utf-8') as f:
                    kept = [json.loads(line)['id'] for line in f]
                cur.execute("SELECT id FROM chunks WHERE id = ANY(%s) AND form = '10-K'", (kept,))
                wanted.update(row[0] for row in cur.fetchall())
            else:
                cur.execute(
                    "SELECT id FROM chunks WHERE form = '10-K' AND id != ALL(%s) ORDER BY random() LIMIT %s",
                    (list(wanted), TARGET_TOTAL - len(wanted)),
                )
                for row in cur.fetchall():
                    wanted.add(row[0])

            for ticker, form, year, period in QUARTERLY_FILINGS:
                cur.execute('SELECT id FROM chunks WHERE ticker = %s AND form = %s AND fiscal_year = %s AND fiscal_period = %s',
                            (ticker, form, year, period))
                wanted.update(row[0] for row in cur.fetchall())

            cur.execute(
                """SELECT id, ticker, period, chunk_index, text, embedding::text, form, fiscal_year, fiscal_period
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
                'embedding': json.loads(row[5]),
                'form': row[6],
                'fiscal_year': row[7],
                'fiscal_period': row[8],
            }
            f.write(json.dumps(record) + '\n')

        print(f'Wrote {len(rows)} chunks to {FIXTURE_PATH}')


def build_query_vectors() -> None:
    from earnings_rag.embeddings import embed_texts
    questions = load_questions(REPO_ROOT / 'eval' / 'questions.yaml')

    # embed only questions without a stored vector: the old vectors stay byte-identical, so the 10-K recall cannot move
    mapping = {}
    if QUERY_PATH.exists():
        with QUERY_PATH.open(encoding='utf-8') as f:
            mapping = json.load(f)
    questions_text = [q['question'] for q in questions]
    texts = [t for t in dict.fromkeys(questions_text) if t not in mapping]
    if texts:
        for text, vector in zip(texts, embed_texts(texts)):
            mapping[text] = vector
    mapping = {t: mapping[t] for t in questions_text}  # drop vectors of questions that no longer exist

    with QUERY_PATH.open('w', encoding='utf-8') as f:
        json.dump(mapping, f)

    print(f'Wrote {len(mapping)} query vectors to {QUERY_PATH}')

def main() -> None:
    build_fixture()
    build_query_vectors()

if __name__ == '__main__':
    main()