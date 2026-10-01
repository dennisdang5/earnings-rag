## 8-8-2026 - Explicit package list in pyproject.toml
- Flat layout such that setuptools sees data/ and eval/
- Listed earnings_rag explicitly rather than switching to src layout
- Chose flat for fewer directories and pip install -e .

## 8-8-2026 - Chunk size, overlap, top-k in config
- Allows eval harness to sweep them programmatically from config file

## 8-10-2026 - 10-K only
- System utilizes annual reports only

## 8-10-2026 - Utilize reportDate and not filingDate
- Fiscal period is ingested rather than when the company reported due to the fact that companies report on different days

## 8-12-2026 - Tables dropped and kept only prose
- Embeddings struggle to discriminate between numeric values like "26,974" and "13,507" because they are similar semantically
- Flattened tables would look like information without being retrievable thus tables were dropped
- Numeric would belong to text to sql rather than XBRL

# 8-12-2026 - Trimmed to item 1
- Filings follow the same structure with the table of contents at the front which we removed in order to embedd useful information such as business insight
- Page headers, numbers, and footers were removed since they repeat throughout and dilute the chunks they land in

# 8-17-2026 - Front matter trim with fixed-prefix fallback
- Fallback cuts a fixed 6000 chars if no heading matches at all for the first item if it doesn't match preset structure

# 8-18-2026 - Fixed size token chunking (500/50)
- Chose token based over character based because token counts vary with content density, whereas, character windows give inconsistent embedding input

# 8-18-2026 - JSONL for chunk output
- line countable

# 8-19-2026 - Hosted embeddings over local sentence transformers
- Local sentence transformer would add PyTorch to the Docker image which is around 2.5 GB memory required

# 8-25-2026 - Exact brute force search with no vector index
- ~1,8000 rows scans in milliseconds and is exact
- HNSW or IVFFlat are approximate and recall would become a function of index tuning

# 8-25-2026 - Corpus split
- 58% of the corpus is COF, 15% AAPL, 27% NVDA
- Bank 10-Ks are far longer than tech ones
- Eval questions must be spread deliberately across tickers

# 8-27-2026 - Temperature=0 for LLM generation
- Setting temperature=0 allows reproducibility rather than allowing the LLM to have randomness in token selection
- Model takes the most likely token everytime

# 8-27-2026 - Explicit "filings do not address this" instruction
- Verified with an out of corpus question
- Without this rule an LLM will answer from training data or synthesize from marginally related chunks 
- Failure mode would make the RAG untrustworthy

# 8-28-2026 - First eval run: 0.111 recall@5
- Initial questions listed one expected chunk each
- But 10-k risk factors are near identical across filing years so retrieval returned a different year's copy of the correct disclosure and socred as a miss
- Widened expected_chunks to list every year's versions which was verified by reading each
- A chunk is valid ground truth if a reader could answer the question from that chunk alone
- Position of the key phrase doesn't matter it only matters whether the answering substance is present and complete and not truncated mid-disclosure

# 8-28-2026 - Fixed size chunking
- Fixed size chunking cuts through topic boundaries and results into mixed topic chunks retrieved worse than focused ones

# 8-29-2026 - Two kinds of retrieval failure found
- Miss 1 and Miss 3 — wrong section entirely (customer default question)
COF_2023-12-31_0041 explicitly lists why customers default: job loss,
rising debt, inflation outpacing wages, unemployment. Retrieval returned
credit *ratings* and credit *quality indicator* chunks instead. Cause:
"credit" means several different things in a bank filing, and the query
matched the wrong sense.
- Miss 2 — right section, wrong chunk (fair value question)
Four chunks (one per year) contain the answer. Retrieval returned the
chunk immediately before each one, all four times. Cause: consecutive
chunks in the fair value note use nearly identical vocabulary, so their
vectors sit almost on top of each other. The one paragraph that holds
the answer is too small a fraction of a 500-token chunk to move its
vector much.
- This suggests Miss 1 points toward metadata filtering by limiting search by section. For miss 2 we should use smaller chunks
such that smaller token sized chunks can answer the question rather than being averaged away
- Both kept as misses because these were real limitations

# 8-30-2026 - Eval baseline: recall@5 = 0.842 (16/19)
- 19 hand written questions with the ground truth verified by reading every chunk
- Roughly even across NVDA/AAPL/COF despite the corpus being 58% COF
- NVDA and AAPL questions all pass
- Questions missed pertain to COF because these filings were 4x longer and contained narrow repeated vocabulary so any bank query has far more near distance competition

# 9-1-2026 - Chunk size sweep result: 500 tokens is local optimum
- Recall @5 0.737 (250 token chunk size), 0.842 (500 token chunk size), 0.787 (750 token chunk size)
- Anchor matching validated against id matching at 500 with identical results
- 500 token chunk size is a local peak
- 250 token chunk size has a thin context per chunk with weak semantic signal, 4,035 competing at similar distances
- 750 token chunk size has topic dilution such that more unrelated material is averaged into the same vector
- Keeping it at default 500/50 provided to be the local optimum

# 9-10-2026 - False refusal caused by the escape hatch wording
- The China export-controls question refused despite retrieving five
on-point chunks at 0.33-0.37 distance. Isolated by calling the model
with the same context and a bare prompt — it answered fine. So the
refusal instruction was the trigger, not retrieval or formatting.
- Cause: "if the context does not contain the answer" reads strictly. The
export-control chunks discuss risks in hedged language ("may", "could"),
and the model apparently judged that insufficient to constitute an
answer.
- Fix: reworded to permit partial and hedged answers, reserving refusal
for context that is genuinely about a different subject. China now
answers; Tesla still refuses.
- Debugging note: isolate by narrowing the call path. build_context was
fine (14,312 chars), PROMPT.format was fine (14,684), generate() still
refused — so the difference had to be in the prompt text itself.

# 9-14-2026 - CI fixture corpus: 200 chunks and reproduces full-corpus recall
- CI can't embed 1,799 chunks per push due to cost, time, and an API key in secrets
- Instead, a committed fixture of ~200 chunks with precomputed embeddings loaded into postgres at CI start
- Fixtures includes all 15 chunks that currently outrank ground truth in the three known misses, plus random distractors
- Results: 0.842 on the fixture which was identical to the full corpus and misses on the same three
- Goal: catches regressions such that a chunking or prompt change that breaks retrieval
- Does not detect drift

# 9-21-2026 - Refusal precision check
- Added a generation stage metric alongside recall@5
- Recall only measures retrieval and it can't see a model that receives the right chunk and refuses anyway

# 9-22-2026 - Generation refusal is sensitive to exact question wording
- "What limits NVIDIA from selling to China?" refused 8/10
- Changing "NVIDIA" to "Nvidia" or "China" to "china" resulted in a 0/10 refusal
- Generation fragility such that the refusal decision sits near a boundary and perhaps a token level difference in the question tip it
- 
# 9-30-2026 - Hand-rolled agent loop alongside the fixed pipeline
- The model chooses when to call tools; the loop (earnings_rag/agent/loop.py) is ~40 lines on the OpenAI SDK with no framework so every message and stop condition is visible
- /ask and the CI recall gate are untouched; the agent is additive
- 10-Qs rather than call transcripts for quarterly text since transcripts aren't on EDGAR, so search_transcripts became search_filings

# 9-30-2026 - Agent tool design
- One pydantic model per tool both generates the JSON schema sent to the model and validates the arguments it sends back
- Tool errors (bad arguments, unknown tool, exceptions) are returned to the model as {"error": ...} rather than raised, so it can retry and one bad call doesn't kill the request
- Budget exhausted forces a final answer (tool_choice="none", truncated=True) rather than an error since a partial answer is still useful
- Citations are [chunk_id] not [n] because positional numbers collide once the agent can run two searches
- search_filings disables keyword company routing (route=False); the model names the company explicitly, so routing must not second-guess it
- period/form are left out of the tool schema until 10-Qs exist; a parameter that silently does nothing makes the model believe it filtered
- Loop is tested with a scripted fake client so CI stays free of API keys

# 9-30-2026 - Agent follow-ups: relevance/scope prompt, novelty note, cutoff + token tracking
- Answer drift is a relevance failure, not a groundedness one: the Apple/Capital One competition answer cited real passages, but about IP, regulation and brand rather than competition. Step 6 should score the two separately
- Scope and relevance rules added to the system prompt. Live result: Tesla went from 4 searches to 1 (it still searched once instead of refusing outright, so the scope rule is followed loosely), but competition drift persisted (IP and brand/ESG items still listed). Prompt wording alone did not fix it; candidates are a reranker or a distance threshold
- Novelty note is advisory: repeats are only known after the search has run, so the loop annotates the result ("k of n passages already returned") and the model may still search again; only agent_max_steps hard-stops it. Judged by returned chunk ids, not an argument cache, because the model rewords queries. Not exercised by the live runs (no repeats occurred), only by unit tests
- AgentResult records cut_off (final answer finish_reason == "length") and per-call usage (input/output tokens, finish_reason), with totals derived from that list. New agent_max_tokens (default 500, separate from llm_max_tokens) so max_tokens can be swept against quality and token cost in step 6 without touching /ask
- Input tokens grow faster than step count because the whole history is re-sent each call; track tokens, not just steps, when testing whether more steps help
- Live runs on the current prompt showed no cutoffs (competition answer used 341 output tokens of 500); the earlier cut-off answer was longer because it padded to several points per company

# 9-30-2026 - calculate tool: AST allowlist, floats, raw arithmetic only
- The model's expression is untrusted input, so evaluation walks the parsed AST and accepts only numbers, + - * / ** and unary +/-. Names, calls, attributes and subscripts are rejected by construction. eval() with empty globals is escapable through attribute chains on literals; a regex can't handle precedence and parentheses
- Every number is converted to float before evaluating. With Python ints, 9**9**9**9 would build an enormous integer and hang the process; with floats it overflows immediately and becomes an error the model can read. Floats are exact to ~9e15, far beyond the dollar figures here
- Expressions are capped at 200 characters, and non-finite or complex results (e.g. (-8)**0.5) are errors
- Errors are written for the model: a comma in 26,974 and a % sign each get a specific hint, because those are the mistakes it makes
- Raw arithmetic only, no pct_change/margin helpers: smallest attack surface and simplest to review. The tool guarantees the arithmetic, not the formula, so a wrong formula (dividing by the wrong year) is still possible; step 6 should measure it
- Live run: given figures in the question, the model called calculate and converted units correctly (125.88%). The search half of that answer drifted into revenue-recognition boilerplate instead of growth drivers, a retrieval problem, not a calculate one
- Live run: the Tesla question now refuses with zero tool calls, but runs vary even at temperature 0, so one run is not evidence the prompt change caused it

# 9-30-2026 - XBRL facts: derive periods from dates, never EPS, and a per-share hazard
- companyfacts repeats every value in later filings as a prior-period comparison, and its fy/fp fields describe the filing that reported the fact, not the period. Apple's quarter ended 2024-03-30 appears as fy=2024 and again as fy=2025. Periods are therefore labeled from each fact's own start/end dates, deduplicated to the latest filing
- Fiscal year = the calendar year it ends in (NVIDIA fiscal 2025 ended 2025-01-26), matching the filings and the 10-K chunk ids
- No company reports Q4 on its own. Q4 = FY - 9-month YTD, Q2/Q3 from YTD totals where needed (cash flow is YTD-only). Derived rows are flagged and cite the filing they came from. Check: NVIDIA FY2025 Q4 = 130,497 - 91,166 = 39,331M, matching its reported Q4
- EPS is never derived: share counts change, so FY EPS minus 9-month EPS is not Q4 EPS
- The obvious revenue concept is wrong for Capital One: RevenueFromContractWithCustomer is only its ~$5.9B of fee revenue, while Revenues is $39.1B (net interest 31.2B + noninterest 7.9B). METRICS is a hand-verified per-company map rather than one standard concept
- Only consolidated totals are in companyfacts; segment data (e.g. NVIDIA Data Center) comes from the inline XBRL in the 10-K files in a later PR. The facts table has a segment column (empty = consolidated) so that needs no migration
- Verified against the live API after ingest: NVDA FY2024/FY2025 revenue 60.922B/130.497B, AAPL FY2024 391.035B, COF FY2024 39.112B, no derived EPS rows, no missing quarters for fiscal years with an annual figure
- HAZARD found during verification: per-share values are on mixed share bases. NVIDIA FY2022 EPS is 3.85 (pre 10:1 split) but FY2023 is 0.17 (post-split), and FY2024 Q1 is 0.82 (pre-split) while Q2/Q3 are adjusted. A period is only re-presented in filings for about two years, so "latest filing wins" does not guarantee split-adjusted values. An EPS growth question across that boundary would give -95%. This table is not exposed to the agent yet; the tool PR must decide how to handle it
- Some FY2020 10-Ks still carried a reported Q4 EPS fact, so two direct (derived=False) Q4 EPS rows exist. The invariant is "no derived EPS", not "no Q4 EPS"

# 9-30-2026 - Split-adjust per-share values (resolves the hazard above)
- Chose to adjust rather than hide EPS or attach a warning: a warning relies on the model heeding it and it can't tell where a split fell, so cross-split comparisons would still be wrong
- Ratios come from XBRL (StockholdersEquityNoteStockSplitConversionRatio1): NVDA 4 and 10, AAPL 7 and 4, COF none. The period dates attached to them are not effective dates (NVIDIA's 10:1 carries 2024-05-31 and 2024-06-30), but the first filing to report a ratio is the first filed after the split, so a split's cutoff is the earliest filing date in its cluster. Cutoffs found: NVDA 2021-08-20 and 2024-08-28, AAPL 2014-07-23 and 2020-10-30
- A value is divided by the product of the ratios of all splits whose cutoff is after its filing date. A filing on the cutoff date reports the ratio, so it is already post-split
- Rows record split_factor (value * split_factor = as reported), so adjustments are visible and reversible. It is the first schema migration: ALTER TABLE ... ADD COLUMN IF NOT EXISTS runs in init_schema, so it works on fresh and existing databases
- Independent check, run before merging: any period reported by more than one filing should agree once all readings are on today's basis. 121 periods across the three tickers, 0 mismatches, worst spread 0.005 (2-decimal EPS rounding; a wrong cutoff would show a 4x or 10x spread). NVIDIA FY2022 to FY2023 EPS change went from -95% to -55.8%
- Failure mode: a company that does not tag the ratio in its first post-split filing gets a cutoff that is too late, double-adjusting values filed in the gap. Run the overlap check for any new ticker
- Precision: EPS is reported to 2 decimals, so a restated 0.17 is coarse (the true value is about 0.174). Net income divided by shares would be finer but has the same share-basis problem
