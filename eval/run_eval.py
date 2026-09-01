import yaml
from pathlib import Path
from earnings_rag.pipeline import retrieve
from earnings_rag.config import REPO_ROOT
from earnings_rag.store import connect

def load_questions(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        return yaml.safe_load(f)

def score(questions: list[dict], k: int = 5) -> dict:
    hits = 0
    scored = 0
    misses = []

    for q in questions:
        expected = set(q.get('expected_chunks') or [])
        if not expected:
            continue # Refusal questions aren't scored on recall

        results = retrieve(q['question'], k=k)
        retrieved = set()
        for r in results:
            retrieved.add(r['id'])

        scored += 1
        if expected & retrieved:
            hits += 1
        else:
            misses.append((q['question'], sorted(expected), sorted(retrieved)))

    return {'recall_at_k': hits / scored, 'hits': hits, 'scored': scored, 'misses': misses}

def validate_questions(questions: list[dict]) -> list[str]:
    """
    Return any expected_chunks ids that don't exist in the database.
    """
    wanted = set()
    for q in questions:
        for chunk_id in (q.get('expected_chunks') or []):
            wanted.add(chunk_id)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id FROM chunks WHERE id = ANY(%s)', (list(wanted),))
            rows = cur.fetchall()

    found = set()
    for row in rows:
        found.add(row[0])

    return sorted(wanted - found)

def validate_anchors(questions: list[dict]) -> None:
    """
    Report which listed chunks each anchor does and doesn't match

    An anchor that matches no chunks is a typo or a paraphrase. Furthermore, an anchor matching only some chunks
    means those years word the disclosure differently.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            for q in questions:
                expected = q.get('expected_chunks') or []
                anchors = q.get('anchors') or []

                if not anchors:
                    print(f'NO ANCHORS: {q["question"][:60]}')
                    continue

                covered = set()
                for anchor in anchors:
                    cur.execute(
                        'SELECT id FROM chunks WHERE id = ANY(%s) AND text ILIKE %s',
                        (expected, f'%{anchor}%'),
                    )
                    rows = cur.fetchall()

                    matched = set()
                    for row in rows:
                        matched.add(row[0])

                    covered = covered | matched

                    if not matched:
                        print(f'    DEAD ANCHOR: {anchor!r}')

                uncovered = set(expected) - covered
                if uncovered:
                    print(f'{q['question'][:60]}')
                    print(f'    Not matched by any anchor: {sorted(uncovered)}')

if __name__ == '__main__':
    questions = load_questions(REPO_ROOT / 'eval' / 'questions.yaml')
    missing = validate_questions(questions)

    if missing:
        print('BAD IDS IN questions.yaml:')
        for chunk_id in missing:
            print(f'{chunk_id}')
        raise SystemExit(1)

    validate_anchors(questions)
    result = score(questions)

    print(f'recall@5: {result["recall_at_k"]:.3f} ({result["hits"]}/{result["scored"]})')
    for question, expected, retrieved in result["misses"]:
        print(f'\nMISS: {question}')
        print(f'expected: {expected}')
        print(f'got:{retrieved}')