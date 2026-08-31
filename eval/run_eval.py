import yaml
from pathlib import Path
from earnings_rag.pipeline import retrieve
from earnings_rag.config import REPO_ROOT

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

if __name__ == '__main__':
    questions = load_questions(REPO_ROOT / 'eval' / 'questions.yaml')
    result = score(questions)

    print(f'recall@5: {result['recall_at_k']:.3f} ({result['hits']}/{result['scored']})')
    for question, expected, retrieved in result["misses"]:
        print(f'\nMISS: {question}')
        print(f'expected: {expected}')
        print(f'got:{retrieved}')