# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A RAG system over SEC 10-K filings for three tickers (NVDA, AAPL, COF): EDGAR ingestion → HTML cleaning → token chunking → OpenAI embeddings → Postgres/pgvector → grounded `gpt-4o-mini` answers with `[n]` citations, served by FastAPI with a streaming single-page UI (`earnings_rag/static/index.html`). `DECISIONS.md` records the reasoning and measurements behind nearly every design choice; check it before changing chunking, retrieval, or the prompt, and add an entry when making a comparable decision.

It is being extended into a **tool-using agent** (see "Project direction and status" below). The original fixed `/ask` pipeline is the baseline and must keep working unchanged.

## Commands

```bash
pip install -r requirements.txt && pip install -e .
docker compose up -d db                  # pgvector on localhost:5433 (matches .env.example)

pytest                                   # all tests
pytest tests/test_is_noise.py::test_is_noise # single test

uvicorn earnings_rag.api:app --reload    # UI at :8000, API docs at :8000/docs
python ask.py "question" [--sources]     # CLI query (fixed pipeline)
python -m earnings_rag.agent "question"  # tool-using agent, prints tool trace then answer

# Build corpus (needs SEC_USER_AGENT, EMBEDDING_API_KEY in .env)
python -c "from earnings_rag.ingest import ingest_all; ingest_all()"   # -> data/raw/<TICKER>/<period>.html, cached
python -c "from earnings_rag.chunking import chunk_all; chunk_all()"   # -> data/chunks_<size>_<overlap>.jsonl
python earnings_rag/pipeline.py                                        # init schema + embed + upsert
python -m earnings_rag.xbrl [--refresh]                                # SEC XBRL facts -> facts table (cached in data/xbrl/)

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
- **Agent** (`earnings_rag/agent/`): `loop.py:run_agent` is the whole loop (call model → run requested tools → append results → repeat, up to `agent_max_steps`, then a forced tool-free answer with `truncated=True`). `AgentResult` also carries `cut_off` (answer hit `agent_max_tokens`) and per-call token `usage`; search results get an advisory note when passages were already returned (`annotate_novelty`). Tools in `tools.py` are a pydantic args model + function; `Tool.call` never raises, it returns `{"error": ...}` to the model. The client is injected, so `tests/test_agent_loop.py` uses a scripted fake. Answers cite `[chunk_id]` or a fact id. `citations.py:check_citations` finds figures stated without an `[id]` and cited ids no tool returned; `run_agent` sends such an answer back once (tools allowed, counts against the step budget) and records `revised`, `uncited` and `unknown_ids` on the result. It checks that a citation exists, not that it supports the claim. The fixed `/ask` pipeline is separate and unchanged. `calculate` (`earnings_rag/calc.py`) evaluates arithmetic through an AST allowlist, never `eval`, with all values as floats so huge powers overflow instead of hanging. Tools registered today: `search_filings`, `get_financials` (with `breakdown`), `calculate` (`DEFAULT_TOOLS` in `tools.py`). `get_financials(company, metric, fiscal_year, period)` reads the `facts` table: ids like `NVDA_revenue_FY2025Q4`, dollars shown in USD millions, 8 rows by default, split-adjusted per-share rows also carry `as_reported`; a metric a company does not report returns an error listing what it does report. `breakdown` ('product' | 'segment' | 'geography') returns every slice on that axis (latest two years by default, annual only), ids like `NVDA_revenue_FY2025_product_DataCenter`; balance metrics (`loans`, `deposits`, `balance=True` in `METRICS`) are values on a date: one-day periods labelled FY/Q1-Q3 by their date, never derived, shown with `balance` and `as_of`; `segments.find_parts` marks overlapping slices with `part_of`, and each result lists the slice names on the other axes so the model can switch axis.
- **XBRL facts** (`xbrl.py`, `facts` table): structured figures from the SEC companyfacts API. `normalize()` is a pure function: it labels periods from each fact's own dates (not the filing's `fy`/`fp`), dedups to the latest filing, and derives missing quarters by subtracting year-to-date totals (`derived=True`; never for EPS). Metric→concept mapping is per company in `METRICS`, hand-verified. `fetch_companyfacts` imports `ingest` lazily because `ingest` raises at import when `SEC_USER_AGENT` is unset (CI). Per-share values (EPS) are rescaled to today's share count: `find_splits` detects splits from jumps in the cover-page share count (the cutoff is the first filing after the jump, corroborated against restated diluted-share readings; `SPLIT_OVERRIDES` covers multi-class issuers whose cover counts are missing), and `split_factor` records the adjustment (`value * split_factor` is the as-reported figure). `ingest_facts` runs `check_split_consistency` and refuses to store a ticker whose readings disagree by a split-sized ratio. Do not use tagged split ratios: they are late, early or duplicated for some companies (see `DECISIONS.md`).
- **Schema changes** use idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` lines in `SCHEMA_SQL` (see `facts.split_factor`), so `init_schema()` migrates existing databases and creates fresh ones.
- **Facts conventions**: fiscal year = the calendar year the fiscal year ends in (NVIDIA fiscal 2025 ended 2025-01-26), matching the 10-K chunk ids. `facts.segment = ''` means consolidated (the column exists for the segment work). `get_facts` returns the latest periods first when no fiscal year is given. Only consolidated totals come from the companyfacts API; segment data lives in the inline XBRL of the 10-K HTML already in `data/raw/`, parsed by `segments.py` into the same table with `axis` = `product` / `segment` / `geography` (`axis = ''` and `segment = ''` are consolidated; `get_facts` defaults to consolidated). `parse_inline` is pure; `ingest_segments` (run by `python -m earnings_rag.xbrl`) refuses to store a ticker whose parsed consolidated values disagree with the API, and prints a report-only sum check per breakdown. A business-segment breakdown with no reported corporate item gets a `derived=True` "Corporate and other" row (consolidated total minus segments); `replace_segment_facts` swaps a ticker's breakdown rows in one transaction so rows the parser stops producing do not linger. Exposed through `get_financials(..., breakdown=...)`. A breakdown for a year comes whole from the latest filing that reports it (never mixed slice by slice). `data/` is git-ignored.
- **Deployment**: Docker image (tagged with git SHA) on AWS ECS Fargate with RDS pgvector; set `DB_SSLMODE=require` for RDS, `disable` locally/CI.

## Project direction and status

**Goal:** an agent whose tools are `search_filings` (pgvector retrieval, now optional per question), `get_financials` (structured XBRL numbers), `calculate` (so the model never does arithmetic), and `compare` (companies or periods, only kept if the eval shows it helps). Quarterly text comes from 10-Qs, not earnings-call transcripts, which are not on EDGAR. The agent loop is hand-rolled on the OpenAI SDK, no framework, so every message and stop condition stays visible.

**Done and merged to main:**
1. Agent loop + `search_filings`. When the step budget runs out the loop forces a tool-free final answer and sets `truncated=True`, since a partial answer is more useful than an error. Citations are `[chunk_id]`. `period`/`form` are left out of the tool schema until 10-Qs exist, because a parameter that silently does nothing makes the model believe it filtered.
2. Prompt/loop follow-ups: corpus-scope and relevance rules in the system prompt, an advisory novelty note on repeated search results (it does not block repeat searches), `cut_off` detection (final answer hit `finish_reason == "length"`), per-call token `usage`, and a separate `agent_max_tokens` (500) so max tokens can be swept against quality and cost.
3. `calculate` tool.
4. XBRL facts: `facts` table, `normalize()`, derived Q4, split-adjusted per-share values with split detection from cover-page share counts (tagged split ratios were wrong for WMT, GOOGL and TSLA; see `DECISIONS.md`).

**Next, in order, each its own branch/PR after the previous one merges:**
- (Done: `agent/eps-precision` and `agent/get-financials`, the latter awaiting merge. EPS takes the most precise reported reading across splits and is never computed from net income ÷ shares; see `DECISIONS.md`.)
- (Done: `agent/xbrl-segments`. Done: `agent/segments-tool`, the `breakdown` argument; see `DECISIONS.md`.)
- (Done: `agent/segment-metrics`: cost of revenue, pre-tax income, provision, non-interest expense, loans and deposits, incl. balances.)
- **NEXT: `agent/metric-resolver`, design note first.** `METRICS` is hand-verified per company and will not scale to many tickers. Planned: candidate concepts per metric, chosen per company by accounting identities (revenue >= its parts, gross profit = revenue - cost, pre-tax - tax = net income) with a printed override table; it must reproduce today's NVDA/AAPL/COF map exactly. See `DECISIONS.md` (segment-metrics entry).
- Then: 10-Q ingestion (a `form` column, fiscal quarter mapping, enabling `period`/`form` filters in `store.search`), `compare`, the agent eval, and `/agent/stream` plus UI.

**Agent eval (planned design):** repeated runs per question, because runs vary even at `temperature=0` (the Tesla question took 4 model calls once and 6 the next). Score groundedness (the cited passage supports the claim) separately from relevance (the claim answers the question). Record model calls, tool calls and input/output tokens: input tokens grow faster than step count because the whole history is re-sent each call. Test whether more steps or a larger `agent_max_tokens` buy better answers at what token cost. CI should use a fake client or recorded traces; live runs are manual like `--refusals`.

**Known issues and open findings:**
- Answer drift: "How do Apple and Capital One each describe their competition risks?" still lists IP and brand/ESG items as competition risks. The prompt rule did not fix it; candidates are a reranker or a distance threshold.
- Miscited derived numbers: a citation can exist and not support the claim ("114% [NVDA_revenue_FY2025]"; a margin percentage citing a gross-profit dollar fact). The citation check cannot see this; it needs the groundedness score in the agent eval.
- The scope rule is followed loosely: the Tesla question sometimes still searches once before refusing.
- `METRICS` in `xbrl.py` is hand-verified per company, so a new ticker needs its XBRL concepts checked (COF "revenue" is `Revenues`; the standard `RevenueFromContract...` concept is only its ~$5.9B of fee revenue). Split detection and its consistency check do scale; metric mapping does not yet.
- GOOGL-style multi-class issuers have no cover-page share count in the API, so splits are not detected for them; the consistency check will refuse to store them until `SPLIT_OVERRIDES` supplies cutoffs. Run ingestion for a new ticker and read the "split check" line: zero split-sized mismatches is the evidence the adjustment is right.
- A real split can be wrongly rejected by corroboration if no period is re-presented between the filings around it; the consistency check should catch it.

## Working agreement

The user is a CS master's student targeting ML/AI engineering roles and is using this project to learn how agents work under the hood and how industry teams work. So:
- Work in **small PRs (~100-200 lines)** on branches named `agent/<step>`. Give a short design note (the problem, the interface, the trade-offs) before coding, and wait for the user's decisions on genuinely open choices. No large unexplained code dumps; explain the mechanics.
- **Opus plans, Sonnet implements** (`/model opusplan`): design in plan mode, then implement after the user approves.
- Every PR adds a `DECISIONS.md` entry in the existing dated style (what was decided, why, and what was measured), updates this file if the architecture changed, and keeps CI hermetic (fake LLM client, no network, no API keys).
- Verify against real data, not only synthetic tests: run the agent live with the DB up, spot-check ingested numbers against known figures, and report what did not work as plainly as what did. Several design errors here were caught only by testing on companies beyond the three in the corpus.
- The existing `/ask` pipeline and its CI recall gate (≥0.75 on the offline fixture) must not regress.
- The user wants designs that scale to many tickers: prefer mechanisms that detect and validate over hand-maintained tables.

## Environment notes

- Windows. Use `.venv/Scripts/python`. LF/CRLF warnings from git are harmless.
- `gh` is installed (`C:\Program Files\GitHub CLI\gh`) and logged in as `dennisdang5`: push the branch and open the PR with `gh pr create`. If `gh auth status` fails, ask the user to run `! gh auth login`. Merges so far were done locally with `git merge --no-ff` after the user said to merge, then pushed to `main` and the branch deleted; only merge after the user says so.
- Docker must be running for anything touching the database (`docker compose up -d db`); the corpus lives in the `earnings` database on port 5433. After pulling, run `python -m earnings_rag.xbrl` to populate or refresh the `facts` table.
- Bash heredocs containing triple quotes failed in this environment; use the Write/Edit tools for multi-line file edits.
- `ingest.py` raises at import time if `SEC_USER_AGENT` is unset, so anything CI imports must not import it at module level.
