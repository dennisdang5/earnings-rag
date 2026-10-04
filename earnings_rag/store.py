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

-- form: '10-K' or '10-Q'. fiscal_year / fiscal_period use the facts table's labels ('FY', 'Q1'-'Q3'), so a quarter's text
-- lines up with its numbers. These ALTERs migrate databases created before the columns existed.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS form TEXT NOT NULL DEFAULT '10-K';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fiscal_year INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fiscal_period TEXT;
-- existing rows are all 10-Ks: their period is the fiscal year end, and the fiscal year is the year it ends in
UPDATE chunks SET fiscal_year = EXTRACT(YEAR FROM period::date)::int, fiscal_period = 'FY'
    WHERE form = '10-K' AND fiscal_year IS NULL AND period LIKE '____-__-__';
CREATE INDEX IF NOT EXISTS chunks_ticker_form_idx ON chunks (ticker, form, fiscal_year);

-- segment '' means consolidated; per-segment rows share the table. axis says which breakdown a segment is on:
-- 'product', 'segment' (business segment) or 'geography'; '' for consolidated rows
CREATE TABLE IF NOT EXISTS facts(
    ticker  TEXT    NOT NULL,
    metric  TEXT    NOT NULL,
    segment TEXT    NOT NULL DEFAULT '',
    axis    TEXT    NOT NULL DEFAULT '',
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
    PRIMARY KEY (ticker, metric, axis, segment, fiscal_year, fiscal_period)
    );

-- value * split_factor is the as-reported figure. This line migrates databases created before the column existed.
ALTER TABLE facts ADD COLUMN IF NOT EXISTS split_factor DOUBLE PRECISION NOT NULL DEFAULT 1;

-- Same for axis, which also joins the primary key (one segment name can sit on two axes). Re-keys only if needed.
ALTER TABLE facts ADD COLUMN IF NOT EXISTS axis TEXT NOT NULL DEFAULT '';
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                   WHERE i.indrelid = 'facts'::regclass AND i.indisprimary AND a.attname = 'axis') THEN
        ALTER TABLE facts DROP CONSTRAINT facts_pkey;
        ALTER TABLE facts ADD PRIMARY KEY (ticker, metric, axis, segment, fiscal_year, fiscal_period);
    END IF;
END $$;
"""

def connect() -> psycopg.Connection:
    return psycopg.connect(settings.db_url)

def upsert_chunks(records: list[dict], vectors: list[list[float]]) -> None:
    params = []
    for record, vector in zip(records, vectors):
        params.append((record['id'], record['ticker'], record['period'],
                       record['chunk_index'], record['text'], str(vector),
                       record.get('form', '10-K'), record.get('fiscal_year'), record.get('fiscal_period')))

    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO CHUNKS (id, ticker, period, chunk_index, text, embedding, form, fiscal_year, fiscal_period)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    form = EXCLUDED.form,
                    fiscal_year = EXCLUDED.fiscal_year,
                    fiscal_period = EXCLUDED.fiscal_period
                """,
                params,
            )
        conn.commit()

def chunk_ids() -> set[str]:
    """Every stored chunk id, so an index run can embed only what is new."""
    with connect() as conn:
        return {r[0] for r in conn.execute('SELECT id FROM chunks').fetchall()}

def _where(ticker: str | None, form: str | None, fiscal_year: int | None, fiscal_period: str | None,
           latest: bool = False) -> tuple[str, list]:
    """
    The WHERE clause and parameters for a filtered search; a filter left as None does not restrict.
    latest keeps only each company's most recent filing (among the given form, if any). Periods are ISO dates, so the
    greatest string is the latest. Computed per company, so a search over all companies gets each one's own latest.
    """
    clauses, params = [], []
    for column, value in (('ticker', ticker), ('form', form), ('fiscal_year', fiscal_year),
                          ('fiscal_period', fiscal_period)):
        if value is not None:
            clauses.append(f'{column} = %s')
            params.append(value)
    if latest:
        same_form = ' AND c2.form = %s' if form is not None else ''
        clauses.append(f'period = (SELECT max(c2.period) FROM chunks c2 WHERE c2.ticker = chunks.ticker{same_form})')
        if form is not None:
            params.append(form)
    return (' WHERE ' + ' AND '.join(clauses)) if clauses else '', params

def text_periods(ticker: str | None = None) -> list[dict]:
    """The filings that have text, oldest first: one row per (ticker, form, period) with its fiscal label."""
    sql = ('SELECT DISTINCT ticker, form, period, fiscal_year, fiscal_period FROM chunks'
           + (' WHERE ticker = %s' if ticker else '') + ' ORDER BY ticker, period')
    with connect() as conn:
        rows = conn.execute(sql, (ticker,) if ticker else ()).fetchall()
    return [dict(zip(('ticker', 'form', 'period', 'fiscal_year', 'fiscal_period'), r)) for r in rows]

def search(query_vector: list[float], k: int = 5, ticker: str | None = None, form: str | None = None,
           fiscal_year: int | None = None, fiscal_period: str | None = None, latest: bool = False) -> list[dict]:
    """
    Exact nearest neighbours by cosine distance among the chunks matching the filters. A filter left as None does not
    restrict: form=None searches 10-Ks and 10-Qs together (pipeline.retrieve defaults to '10-K' so the fixed /ask
    pipeline is unchanged).
    """
    vector = str(query_vector)
    where, params = _where(ticker, form, fiscal_year, fiscal_period, latest)
    sql = f"""
        SELECT id, ticker, period, text, embedding <=> %s::vector AS distance, form, fiscal_year, fiscal_period
        FROM chunks{where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [vector, *params, vector, k])
            rows = cur.fetchall()

    results = []
    for row in rows:
        results.append({
            'id': row[0],
            'ticker': row[1],
            'period': row[2],
            'text': row[3],
            'distance': row[4],
            'form': row[5],
            'fiscal_year': row[6],
            'fiscal_period': row[7],
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

FACT_COLUMNS = ['ticker', 'metric', 'segment', 'axis', 'fiscal_year', 'fiscal_period', 'concept', 'unit', 'value',
                'period_start', 'period_end', 'derived', 'form', 'accession', 'filed', 'split_factor']

UPSERT_FACTS_SQL = f"""
    INSERT INTO facts ({', '.join(FACT_COLUMNS)})
    VALUES ({', '.join(['%s'] * len(FACT_COLUMNS))})
    ON CONFLICT (ticker, metric, axis, segment, fiscal_year, fiscal_period) DO UPDATE SET
        {', '.join(f'{c} = EXCLUDED.{c}' for c in FACT_COLUMNS[6:])}
"""

def upsert_facts(rows: list[dict]) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_FACTS_SQL, [tuple(r[c] for c in FACT_COLUMNS) for r in rows])
        conn.commit()

def replace_segment_facts(ticker: str, rows: list[dict]) -> None:
    """
    Swap a ticker's breakdown rows (axis != '') for these, in one transaction. An upsert alone leaves stale rows behind
    when the parser stops producing one (a corrected rule, a dropped bad fact), so the old set is deleted first.
    Consolidated rows are not touched.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM facts WHERE ticker = %s AND axis <> ''", (ticker,))
            cur.executemany(UPSERT_FACTS_SQL, [tuple(r[c] for c in FACT_COLUMNS) for r in rows])
        conn.commit()

def get_facts(ticker: str, metric: str, fiscal_year: int | None = None, fiscal_period: str | None = None,
              segment: str = '', axis: str = '', limit: int = 12) -> list[dict]:
    """
    Most recent first. With no fiscal_year, this returns the latest periods, which answers "last quarter" questions.
    """
    sql = f'SELECT {", ".join(FACT_COLUMNS)} FROM facts WHERE ticker = %s AND metric = %s AND segment = %s AND axis = %s'
    params = [ticker, metric, segment, axis]
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

def fact_years(ticker: str, metric: str, segment: str = '') -> tuple[int, int] | None:
    """(first, last) fiscal year stored for a metric, or None if there is none."""
    with connect() as conn:
        row = conn.execute('SELECT min(fiscal_year), max(fiscal_year) FROM facts '
                           'WHERE ticker = %s AND metric = %s AND segment = %s', (ticker, metric, segment)).fetchone()
    return None if row[0] is None else (row[0], row[1])

def available_metrics(ticker: str) -> list[str]:
    """Metrics with consolidated rows for a company, in METRICS order: what the company reports, as resolved at ingest."""
    from earnings_rag.resolver import METRICS
    with connect() as conn:
        have = {r[0] for r in conn.execute("SELECT DISTINCT metric FROM facts WHERE ticker = %s AND axis = '' "
                                           "AND segment = ''", (ticker,)).fetchall()}
    return [m for m in METRICS if m in have]

def get_breakdown(ticker: str, metric: str, axis: str, fiscal_year: int | None = None, years: int = 2,
                  fiscal_period: str = 'FY') -> list[dict]:
    """
    Every slice on one axis, for the given fiscal year or the latest `years` years. Latest first, largest first. Annual
    (FY) by default: quarterly slices share the table, and mixing them into a year would double-count it.
    """
    sql = (f'SELECT {", ".join(FACT_COLUMNS)} FROM facts WHERE ticker = %s AND metric = %s AND axis = %s '
           'AND fiscal_period = %s')
    params: list = [ticker, metric, axis, fiscal_period]
    if fiscal_year is not None:
        sql += ' AND fiscal_year = %s'
        params.append(fiscal_year)
    else:
        sql += (' AND fiscal_year IN (SELECT DISTINCT fiscal_year FROM facts WHERE ticker = %s AND metric = %s'
                ' AND axis = %s AND fiscal_period = %s ORDER BY fiscal_year DESC LIMIT %s)')
        params += [ticker, metric, axis, fiscal_period, years]
    sql += ' ORDER BY fiscal_year DESC, value DESC'
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(zip(FACT_COLUMNS, row)) for row in cur.fetchall()]

def breakdown_quarters(ticker: str, metric: str, axis: str) -> tuple[str, str] | None:
    """('Q1 FY2023', 'Q2 FY2027'): the first and last quarter a breakdown has slices for, or None."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT fiscal_year, fiscal_period FROM facts WHERE ticker = %s AND metric = %s AND axis = %s "
            "AND fiscal_period IN ('Q1', 'Q2', 'Q3') GROUP BY fiscal_year, fiscal_period ORDER BY fiscal_year, fiscal_period",
            (ticker, metric, axis)).fetchall()
    return (f'{rows[0][1]} FY{rows[0][0]}', f'{rows[-1][1]} FY{rows[-1][0]}') if rows else None

def available_breakdowns(ticker: str, corporate: str) -> list[tuple[str, str, int, int]]:
    """
    (metric, axis, first year, last year) for each breakdown with at least two real slices. A lone corporate row is
    not a breakdown: Apple's R&D has only "Corporate and other", equal to the whole figure, because it is unallocated.
    """
    with connect() as conn:
        return conn.execute(
            "SELECT metric, axis, min(fiscal_year), max(fiscal_year) FROM facts "
            "WHERE ticker = %s AND axis <> '' AND fiscal_period = 'FY' GROUP BY metric, axis "
            "HAVING count(DISTINCT segment) FILTER (WHERE segment <> %s) >= 2 ORDER BY metric, axis",
            (ticker, corporate)).fetchall()

def breakdown_names(ticker: str, metric: str) -> dict[str, list[str]]:
    """axis -> the slice names in its latest year, largest first. Tells the model which axis a name like Data Center is on."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT axis, segment FROM facts f WHERE ticker = %s AND metric = %s AND axis <> '' AND fiscal_period = 'FY' AND fiscal_year = "
            "(SELECT max(fiscal_year) FROM facts WHERE ticker = f.ticker AND metric = f.metric AND axis = f.axis "
            "AND fiscal_period = 'FY') "
            "ORDER BY axis, value DESC", (ticker, metric)).fetchall()
    names: dict[str, list[str]] = {}
    for axis, segment in rows:
        names.setdefault(axis, []).append(segment)
    return names

def init_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()
