# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A RAG system over SEC 10-K filings for three tickers (NVDA, AAPL, COF): EDGAR ingestion → HTML cleaning → token chunking → OpenAI embeddings → Postgres/pgvector → grounded `gpt-4o-mini` answers with `[n]` citations, served by FastAPI with a streaming single-page UI (`earnings_rag/static/index.html`). `DECISIONS.md` records the reasoning and measurements behind nearly every design choice; check it before changing chunking, retrieval, or the prompt, and add an entry when making a comparable decision.

## Commands

```bash
pip install -r requirements.txt && pip install -e .
docker compose up -d db                  # pgvector on localhost:5433 (matches .env.example)

pytest                                   # all tests
pytest tests/test_is_noise.py::test_is_noise # single test

uvicorn earnings_rag.api:app --reload    # UI at :8000, API docs at :8000/docs
python ask.py "question" [--sources]     # CLI query

# Build corpus (needs SEC_USER_AGENT, EMBEDDING_API_KEY in .env)
python -c "from earnings_rag.ingest import ingest_all; ingest_all()"   # -> data/raw/<TICKER>/<period>.html, cached
python -c "from earnings_rag.chunking import chunk_all; chunk_all()"   # -> data/chunks_<size>_<overlap>.jsonl
python earnings_rag/pipeline.py                                        # init schema + embed + upsert

# Eval
python eval/run_eval.py --match anchor              # recall@5 on full corpus (embeds questions via API)
python eval/run_eval.py --match anchor --offline    # uses precomputed vectors, as CI does
python eval/run_eval.py --refusals                  # generation check, calls the LLM
# other flags: --match id, --no-route, --min-recall <float>
```

## CI

`.github/workflows/ci.yml` is hermetic (no API keys): it runs `pytest`, loads `eval/fixture_chunks.jsonl` (200 chunks with embeddings) via `eval/load_fixture.py`, then runs `run_eval.py --match anchor --offline --min-recall 0.75`. To reproduce locally, point `DB_NAME` at a database ending in `_ci` — `load_fixture.py` refuses any other name because it `TRUNCATE`s `chunks`.

## Architecture notes

- **All tunables live in `earnings_rag/config.py`** (`Settings`, pydantic-settings reading `.env`): chunk size/overlap, top_k, models, tickers, embedding dim. The eval harness relies on sweeping these. `embedding_dim` must match `embedding_model`, and the pgvector column dimension is baked into the schema at `init_schema()` time.
- **Chunk IDs are positional**: `{TICKER}_{period}_{chunk_index:04d}`, where `period` is the fiscal `reportDate` (not filing date). Any chunking change renumbers them. That is why eval ground truth uses **anchor phrases** (verbatim substrings in `eval/questions.yaml`) as the primary match mode; `expected_chunks` IDs and `COMPETITORS` in `eval/build_fixture.py` are ID-based and go stale if chunking changes — rebuild the fixture (`python eval/build_fixture.py`, needs full corpus + API key) and re-verify.
- **Retrieval** (`pipeline.retrieve` → `store.search`): exact brute-force cosine search (`<=>`), deliberately no HNSW/IVFFlat index. `detect_ticker` routes to a single company's chunks only when exactly one company is named; an explicit `ticker` arg overrides. `retrieve` accepts a precomputed `query_vector`, which is how offline eval avoids the API.
- **Generation** (`llm.py`): temperature 0, single user-message prompt. The refusal string `"The provided filings do not address this."` is matched exactly by `eval/run_eval.py` and the UI, so keep them in sync if the prompt changes. Prompt wording was tuned against false refusals on hedged/risk-framed passages — re-run `--refusals` after edits.
- **Parsing** (`chunking.py`): drops tables entirely (numeric questions are out of scope by design), strips inline-XBRL headers, filters page furniture via `is_noise`, and trims front matter to Item 1 with a fixed 6000-char fallback.
- **API** (`api.py`): `/ask` returns JSON; `/ask/stream` emits SSE events `status` → `sources` → `token`* → `done` (or `error`), consumed by `static/index.html`. The UI builds all model/filing text as DOM text nodes — never use `innerHTML` for it. Prometheus metrics at `/metrics`, structlog JSON logging.
- **Deployment**: Docker image (tagged with git SHA) on AWS ECS Fargate with RDS pgvector; set `DB_SSLMODE=require` for RDS, `disable` locally/CI.
