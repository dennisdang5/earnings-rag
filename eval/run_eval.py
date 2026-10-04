import sys
import json
from earnings_rag.agent.tools import SearchFilingsArgs, search_filters
from earnings_rag.pipeline import retrieve, ask
from earnings_rag.config import REPO_ROOT
from earnings_rag.store import connect
from earnings_rag.questions import load_questions

QUERY_PATH = REPO_ROOT / 'eval' / 'fixture_queries.json'
REFUSAL = 'The provided filings do not address this.'

def load_query_vectors(offline: bool) -> dict:
    """
    Load precomputed question embeddings for offline (CI) runs
    """
    if not QUERY_PATH.exists():
        if offline:
            raise SystemExit('offline mode requires eval/fixture_queries.json')
        return {}
    with QUERY_PATH.open(encoding='utf-8') as f:
        return json.load(f)

def check_anchors_present(questions: list[dict]) -> None:
    """
    Fail if any question lacks anchors
    """
    bad = []
    for q in questions:
        if q.get('expect_refusal'):
            continue
        if not (q.get('anchors') or []):
            bad.append(q['question'])

    if bad:
        print('QUESTIONS WITH NO ANCHORS:')
        for question in bad:
            print(f'    {question}')
        raise SystemExit(1)

def quarterly_search(q: dict, k: int, vector: list[float] | None) -> list[dict]:
    """
    Run a question the way the agent's search_filings does: its `filters` (company, fiscal_year, period, latest) go
    through search_filters, the same mapping the tool uses, then retrieve with keyword routing off.
    """
    args = SearchFilingsArgs(query=q['question'], **q['filters'])
    f = search_filters(args)
    if 'error' in f:
        raise SystemExit(f'bad filters for {q["question"][:60]}: {f["error"]}')
    return retrieve(q['question'], k=k, ticker=args.company, route=False, query_vector=vector, form=f['form'],
                    fiscal_year=args.fiscal_year, fiscal_period=f['fiscal_period'], latest=args.latest)


def wrong_filing(q: dict, results: list[dict]) -> list[str]:
    """Ids of results outside the filing the filters asked for (a filter that did nothing shows up here, not as recall)."""
    filters = q['filters']
    bad = []
    for r in results:
        if filters.get('company') and r['ticker'] != filters['company']:
            bad.append(r['id'])
        elif filters.get('fiscal_year') is not None and r.get('fiscal_year') != filters['fiscal_year']:
            bad.append(r['id'])
        elif filters.get('period') in ('Q1', 'Q2', 'Q3') and (r.get('form'), r.get('fiscal_period')) != ('10-Q', filters['period']):
            bad.append(r['id'])
    return bad


def score(questions: list[dict], k: int = 5, match: str = 'anchor', offline: bool = False, route: bool = True,
          quarterly: bool = False) -> dict:
    """
    Compute recall@k over the question set. quarterly=False scores the questions without `filters` (the 10-K set, whose
    CI gate is unchanged); quarterly=True scores only those with `filters`, searched like the agent's search_filings.
    """
    vectors = load_query_vectors(offline)

    hits = 0
    scored = 0
    misses = []
    misfiled = []

    for q in questions:
        if bool(q.get('filters')) != quarterly:
            continue
        if q.get('expect_refusal'):
            continue # refusal questions have no correct chunk

        expected = set(q.get('expected_chunks') or [])
        anchors = q.get('anchors') or []

        if not expected and not anchors:
            continue # Refusal questions aren't scored on recall

        vector = vectors.get(q['question'])
        if offline and vector is None:
            raise SystemExit(f'No precomputed vector for: {q["question"][:60]}')

        if quarterly:
            results = quarterly_search(q, k, vector)
            misfiled.extend((q['question'], rid) for rid in wrong_filing(q, results))
        else:
            results = retrieve(q['question'], k=k, query_vector=vector, route=route)

        if match == 'anchor':
            is_hit = False
            for r in results:
                text = r['text'].lower()
                for anchor in anchors:
                    if anchor.lower() in text:
                        is_hit = True

        else:
            retrieved = set()
            for r in results:
                retrieved.add(r['id'])
            is_hit = bool(expected & retrieved)

        scored += 1
        if is_hit:
            hits += 1
        else:
            got = []
            for r in results:
                got.append(r['id'])
            misses.append((q['question'], anchors, got))

    return {'recall_at_k': hits / scored if scored else 0.0,
            'hits': hits,
            'scored': scored,
            'misses': misses,
            'misfiled': misfiled,
    }

def score_refusals(questions: list[dict]) -> dict:
    """
    Check generation answers when it should and refuses when it should.
    """
    false_refusals = []      # right chunk retrieved model refused anyway
    missed_refusals = []     # out of corpus model answered anyway
    retrieval_refusals = []  # wrong chunks retrieved model refused (correct)
    checked = 0

    for q in questions:
        expect = bool(q.get('expect_refusal'))
        anchors = q.get('anchors') or []

        if q.get('filters') or (not expect and not anchors):
            continue  # quarterly questions need the agent's filters; the fixed pipeline cannot answer them

        checked += 1
        print(f'  [{checked}] {q["question"][:70]}')

        result = ask(q['question'])
        answer = result['answer']
        refused = REFUSAL.lower() in answer.lower()

        retrieved_ok = False
        for source in result['sources']:
            text = source['text'].lower()
            for anchor in anchors:
                if anchor.lower() in text:
                    retrieved_ok = True

        if expect:
            if not refused:
                missed_refusals.append((q['question'], answer[:200]))
        elif refused:
            if retrieved_ok:
                false_refusals.append(q['question'])
            else:
                retrieval_refusals.append(q['question'])

    return {
        'checked': checked,
        'false_refusals': false_refusals,
        'missed_refusals': missed_refusals,
        'retrieval_refusals': retrieval_refusals,
    }

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
                if q.get('expect_refusal'):
                    continue

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
                    print(f'{q["question"][:60]}')
                    print(f'    Not matched by any anchor: {sorted(uncovered)}')


def run_refusal_check(questions: list[dict]) -> None:
    print('Checking generation (one LLM call per question)...\n')
    result = score_refusals(questions)

    n_false = len(result['false_refusals'])
    n_missed = len(result['missed_refusals'])

    print(f'\nchecked {result["checked"]} questions')
    print(f'false refusals:  {n_false}   (answerable, but the model refused)')
    print(f'missed refusals: {n_missed}   (out of corpus, but the model answered)')

    n_retrieval = len(result['retrieval_refusals'])
    print(f'retrieval refusals: {n_retrieval}   (wrong chunks retrieved; refusal was correct)')

    for question in result['false_refusals']:
        print(f'\nFALSE REFUSAL: {question}')

    for question, answer in result['missed_refusals']:
        print(f'\nMISSED REFUSAL: {question}')
        print(f'  answered: {answer}')

    for question in result['retrieval_refusals']:
        print(f'\nRETRIEVAL REFUSAL: {question}')

    if n_false or n_missed:
        raise SystemExit(1)


def main() -> None:
    questions = load_questions(REPO_ROOT / 'eval' / 'questions.yaml')

    if '--refusals' in sys.argv:
        run_refusal_check(questions)
        return

    match_mode = 'anchor'
    if '--match' in sys.argv:
        match_mode = sys.argv[sys.argv.index('--match') + 1]

    offline = '--offline' in sys.argv

    route = '--no-route' not in sys.argv

    threshold = None
    if '--min-recall' in sys.argv:
        threshold = float(sys.argv[sys.argv.index('--min-recall') + 1])

    quarterly_threshold = None
    if '--min-recall-quarterly' in sys.argv:
        quarterly_threshold = float(sys.argv[sys.argv.index('--min-recall-quarterly') + 1])

    if match_mode == 'anchor':
        check_anchors_present(questions)

    if match_mode == 'id':
        missing = validate_questions(questions)
        if missing:
            print('BAD IDS IN questions.yaml:')
            for chunk_id in missing:
                print(f'  {chunk_id}')
            raise SystemExit(1)
        validate_anchors(questions)

    result = score(questions, match=match_mode, offline=offline, route=route)

    print(f'\nmatch={match_mode} offline={offline} route={route} '
          f'recall@5: {result["recall_at_k"]:.3f} '
          f'({result["hits"]}/{result["scored"]})')

    for question, anchors, got in result['misses']:
        print(f'\nMISS: {question}')
        print(f'  anchors: {anchors}')
        print(f'  got:     {got}')

    failed = False
    if threshold is not None and result['recall_at_k'] < threshold:
        print(f'\nFAIL: recall {result["recall_at_k"]:.3f} below threshold {threshold}')
        failed = True

    if match_mode == 'anchor' and any(q.get('filters') for q in questions):
        qres = score(questions, match=match_mode, offline=offline, route=route, quarterly=True)
        print(f'\nquarterly (filtered like search_filings) recall@5: {qres["recall_at_k"]:.3f} '
              f'({qres["hits"]}/{qres["scored"]})')
        for question, anchors, got in qres['misses']:
            print(f'\nQUARTERLY MISS: {question}')
            print(f'  anchors: {anchors}')
            print(f'  got:     {got}')
        for question, rid in qres['misfiled']:
            print(f'\nWRONG FILING: {rid} for {question}')
        if qres['misfiled']:
            print('FAIL: results outside the requested filing (a filter did nothing)')
            failed = True
        if quarterly_threshold is not None and qres['recall_at_k'] < quarterly_threshold:
            print(f'FAIL: quarterly recall {qres["recall_at_k"]:.3f} below threshold {quarterly_threshold}')
            failed = True

    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()