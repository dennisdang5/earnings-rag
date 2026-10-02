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

# 9-30-2026 - Split detection: cover-page share counts, not tagged ratios (supersedes the cutoff rule above)
- Tested the tag-based cutoff on 8 more companies and it was wrong in three different ways: Walmart's ratio was first tagged in June 2024, three months after its first post-split 10-K (that 10-K and the next 10-Q would be adjusted twice); Alphabet tagged its 20:1 ratio in a pre-split 10-Q (that filing would not be adjusted); and Tesla's tags for one split span more than 180 days, so clustering by date produced duplicate events (an old value would be divided by 225 instead of 15). Verdict: tags cannot be trusted to mark the boundary
- The share count on each filing's cover page (dei:EntityCommonStockSharesOutstanding, one per filing) jumps by the split ratio exactly between the last pre-split and first post-split filing. Detected correctly on NVDA, AAPL, TSLA, AMZN, WMT, AVGO, CMG, NFLX and Citigroup's 1:10 reverse split, with ratios like 4.013x, 9.972x, 20.026x, 49.859x and no false positives among split-like jumps
- A split candidate is a consecutive-filing jump within 6% of a split ratio or its reciprocal. It is then corroborated: a real split restates old periods, so a period read both before and after the cutoff shows diluted shares scaled by the ratio, while an acquisition that doubles the share count restates nothing (the same period reads ~1x) and is rejected. Capital One's Discover deal raised shares 1.67x and is not split-like anyway
- Multi-class issuers (GOOGL) have no cover-page count in this API, so detection finds nothing. SPLIT_OVERRIDES (empty for now) supplies their cutoffs by hand
- Safety net: ingest_facts runs check_split_consistency, which compares every EPS period reported by more than one filing after adjustment, and refuses to store a ticker where more than 5% of periods disagree by a split-sized ratio. A missed split (GOOGL: 58.61 vs 2.93, exactly 20x) shows as 7 of 53 periods (13%) and blocks ingestion
- The first version of that check counted every disagreement and would have blocked Tesla (4 of 52) and Amazon (8 of 110) wrongly, because genuine restatements (Tesla's Q1 2024 EPS 0.34 -> 0.41 for an accounting change) also disagree. Mismatches are now classified: split-sized (ratio >= 1.8 and near a split ratio) count toward the threshold, restatement-sized are only reported. Across 11 tickers, only GOOGL had a meaningful split-sized rate; Amazon had 2 of 110 from one garbled filing
- The rounding tolerance scales with the adjustment: each reading carries 0.005 / split_factor of rounding error, so a reverse split amplifies it (Citigroup -0.80 x 10 = -8.0 against a restated -7.99 is rounding, not an error). Forward splits shrink it
- Adding a ticker: run ingestion and read the split-check line. Zero split-sized mismatches is the evidence; a high rate means a missing or wrong split to override

# 10-1-2026 - EPS precision: take the most precise reported reading, do not compute EPS
- Problem: EPS is published to 2 decimals, so each split divides the precision away. NVDA FY2023 EPS is 1.74 in the filings before the 10:1 split and 0.17 after; "latest filing wins" kept 0.17 (true value about 0.174). Growth maths on the coarse value is off (1.19 / 0.17 = 7.0x vs 1.19 / 0.174 = 6.84x)
- Rejected: net income / split-adjusted diluted shares. It is wrong for Capital One: FY2023 12.75 vs reported 11.95 (preferred dividends and participating securities come out of the numerator). A tolerance guard of 0.005 / split_factor + 0.002 x |eps| would still let a computed 11.97 replace the official 11.95. A computed figure the company never published is a wrong answer if it differs, and the user's priority is answers grounded in what the company reported
- Chosen: for per-share metrics, each period uses the reading filed before the most splits, rescaled (1.74 / 10 = 0.174, the company's own figure). It is accepted only if it agrees with the latest reading within the latest reading's rounding (0.005 / its split_factor); otherwise the latest wins, so genuine restatements (Tesla's Q1 2024 0.34 -> 0.41) are not overridden by a stale reading. Without a split between readings the latest still wins
- No schema change: filed, accession and split_factor already record which filing the value came from
- Measured on the cached SEC data (before vs after): 13 of 85 EPS rows changed, all NVDA and AAPL (e.g. NVDA FY2023 0.17 -> 0.174, FY2024 1.19 -> 1.193, AAPL FY2020 Q1 1.25 -> 1.2475); COF has no splits and is unchanged (FY2023 11.95)
- Cost: precision is capped by the earliest filing's two decimals divided by the split factor

# 10-1-2026 - get_financials tool: expose the facts table to the agent
- Interface: get_financials(company, metric, fiscal_year, period). Company and metric are Literal types built from METRICS, so the model can only name metrics that exist; period is FY/Q1-Q4 (named period, not quarter, because FY is not a quarter). Rows have ids like NVDA_revenue_FY2025Q4 and a {"results": [...]} shape, so annotate_novelty and [id] citations work unchanged
- Dollars are shown in USD millions, as the 10-Ks print them: all 17 USD metric series (about 560 rows) are whole millions, so nothing is rounded, and the model copies 130497 instead of a 12-digit number into calculate
- Split-adjusted rows add split_adjusted and as_reported (EPS 0.174, as_reported 1.74), so an answer can match the filing it cites instead of looking inconsistent with it
- A company/metric pair that does not exist (COF gross_profit) returns an error listing what is available, and the tool description lists metrics per company. An empty result says which fiscal years exist. Default is 8 rows (two years of quarters plus FY rows) because every returned row is re-sent on later model calls
- Verified live (DB up, gpt-4o-mini): NVDA revenue FY2024->FY2025 +114.20% (60,922 -> 130,497, 3 model calls, 4.8k input tokens); NVDA EPS FY2023->FY2024 used 0.174 and 1.193 (+585.63%); AAPL FY2024 gross margin 46.21% (gross profit 180,683 / revenue 391,035); COF gross profit returned the tool's error and the agent refused
- Findings, not fixed here: (1) on the mixed question "Why did NVIDIA's revenue grow so much in fiscal 2025?" the agent called get_financials but then stated "$130.5 billion" and "114%" without a fact-id citation and without calculate (the 114% appears to come from passage text), so the prompt's citation and no-arithmetic rules are followed loosely when text and numbers mix; (2) for COF gross profit it answers with the generic refusal string rather than saying banks do not report gross profit; the tool error carries that information but the prompt tells it to refuse

# 10-1-2026 - get_financials follow-ups: citation and "not reported" prompt rules
- CI failed on the first push: test_get_financials imported tests.test_agent_loop, which only resolves under `python -m pytest` (repo root on sys.path), not the plain `pytest` CI runs. The fake OpenAI client moved to tests/fakes.py, imported by both test files. Run plain `pytest` locally before pushing
- COF gross profit: the agent answered with the generic refusal sentence although the tool's error said the metric is not reported. Added a rule: say the company does not report the metric and list related ones it does. Baseline 0/3 runs correct, after 3/3 ("Capital One does not report gross profit. The available financial metrics ... include ... net interest income, and noninterest income"). Kept generic, not bank-specific, so it scales to new tickers
- Mixed question ("Why did NVIDIA's revenue grow so much in fiscal 2025?"): the headline "$130.5 billion ... 114%" was uncited in 3/3 runs. Added a rule that every number needs a citation right after it (fact id for get_financials values, passage id for figures a passage states). After: cited in 3/3, but with [NVDA_revenue_FY2025] alone, which supports $130.5B and not the 114% change. Open groundedness finding, deferred to the agent eval
- Tried and reverted: a rule that a growth rate must cite every fact it was computed from. It changed the tool path (one search, no get_financials) and the headline lost its citation again in 3/3 runs. Prompt rules interact: each wording change altered the search query and the whole trajectory, so rules must be judged over repeated runs, not one
- Reverting that rule first dropped a newline, merging two rules into one line; runs on that broken prompt varied (one vague answer without drivers). With the newline fixed, the final prompt gave identical runs (3/3 each for both questions). Regression check: revenue growth, EPS change and Apple gross margin still use calculate and cite both facts; the Tesla question still refuses without searching
- Next candidate (own PR): a deterministic post-answer check that finds sentences with numbers but no [id] and gives the model one revision turn

# 10-1-2026 - Citation check: deterministic post-answer check with one revision turn
- Prompt rules about citations are followed loosely and every wording change shifted the whole tool path (see the previous entry), so the check lives in code: agent/citations.py check_citations(answer, seen_ids) returns the sentences that state a figure with no [id] (uncited) and cited ids that no tool returned (unknown_ids; seen_ids is the set the novelty note already tracks)
- A figure is a number with a financial marker ($, %, million/billion/trillion). Years, list markers, ids and product numbers (RTX 40, H100) are not figures. Chosen for precision: a false positive costs a pointless model call, a miss leaves us where we were. Sentences are judged separately; comma-separated brackets like [A, B] count; markdown list items are split
- When the model answers with problems and has not been revised yet, run_agent appends the answer plus a message listing the offending sentences and invalid ids, and continues. Tools stay allowed (it can fetch a missing fact) and the turn counts against the step budget. A second failing answer is accepted and flagged (AgentResult.revised/uncited/unknown_ids, printed by the CLI), so it cannot loop. The forced answer after the budget runs out is checked but not revised
- Measured live (gpt-4o-mini): the mixed question "Why did NVIDIA's revenue grow so much in fiscal 2025?" fired in 2 of 3 CLI runs and 0 of 11 later runs (the model is intermittently nondeterministic at temperature 0). "How much did NVIDIA's net income grow in fiscal 2025 and what drove it?" fired in 3 of 4: the answer gave a gross margin "72.7% to 75.0%" with no citation, the revision fetched gross_profit facts and cited them. Revising roughly doubles input tokens (10.0k -> 21.4k) because the whole history is re-sent; it costs nothing when the answer is already cited. Regression: revenue growth, Apple margin, COF, Tesla and a competition-risk question unchanged; the EPS question was revised once for an uncited growth sentence
- What it does not catch, and the evidence: the citation can exist and not support the claim. The revised net-income answer cites [NVDA_gross_profit_FY2025] for a margin percentage; that fact is gross profit in dollars, so it is the right company and period but not the figure. Same family as "114% [NVDA_revenue_FY2025]". Needs a judge (groundedness in the agent eval), not a regex. One of four runs still ended with an uncited sentence after its revision and was flagged, not looped
- Test change: test_agent_run_through_the_real_calculate_tool scripted the answer "50% growth", which the check correctly sent back; its answer no longer contains a figure because that test is about calculate, not citations

# 10-2-2026 - Segment, product and geographic facts from the 10-K inline XBRL
- Problem: the companyfacts API drops every dimensioned fact, so "NVIDIA Data Center revenue" was answerable only from passage text (tables are not chunked). The breakdowns are tagged in the inline XBRL of the 10-K HTML already in data/raw. segments.py parses them into the facts table (new `axis` column: product / segment / geography; '' = consolidated), FY rows only until 10-Q ingestion
- Probed all three tickers before designing. Three standard axes (ProductOrServiceAxis, StatementBusinessSegmentsAxis, StatementGeographicalAxis) carry everything, so there is one 3-entry axis map and no per-company table. "Single-dimension only" would have been wrong: since ASU 2023-07 business segments carry a second dimension (ConsolidationItemsAxis=OperatingSegmentsMember; Apple's FY2022 10-K has the segment alone, FY2025 both), so that member is a qualifier and is dropped before counting dimensions. Contexts that still have 2+ dimensions (Capital One fees by product by segment) are skipped
- Companies rename members between years (NVDA ComputeAndNetworkingMember -> ...SegmentMember), so names are normalized (strip Member/Segment, split CamelCase) and both spellings are one series. Chosen over the filing's label linkbase to avoid another download and network code; cost is ugly names ("I Phone"). ConsolidationItemsAxis=CorporateNonSegmentMember is stored as "Corporate and other" because segments only reconcile to the total with it (COF revenue: 28,164 + 8,718 + 3,601 - 1,371 = 39,112)
- Axes are hierarchical (Data Center = Compute + Networking; Apple Products = iPhone + Mac + ...), so nothing sums a whole axis blindly
- Schema: `axis` joins the primary key (a name can sit on two axes), added by an idempotent ADD COLUMN plus a guarded DO block that re-keys only if axis is not yet in the key. Ran against the existing database: worked in place
- Validation, in the same detect-and-validate style as the split check. (1) Blocking parser check: every consolidated value read from the HTML must equal the SEC API's value for the same concept and period; this tests scale, sign and number formats against an independent source (NVDA 72, AAPL 72, COF 60 values, 0 mismatches). (2) Report-only sum check: does some subset of a breakdown's members add to the consolidated total (subset, because of the hierarchy). 65 of 69 breakdowns reconcile
- The sum check found a real data problem before merge: NVIDIA's FY2026 10-K tags a second "corporate" operating income equal to the total of the operating segments (139,297; 87,960 for FY2025; the earlier filings give -6,507), and "latest filing wins" let it overwrite the correct figure. Rule: a corporate item equal to the total of the operating segments is a tagging artifact and is dropped. NVDA FY2026 operating income by segment therefore has no corporate row and does not reconcile (reported, not hidden)
- The other misses are expected: Apple's FY2020-2022 operating income by segment does not reconcile because unallocated R&D and corporate costs are not tagged to a member in those filings. Report-only for this reason: a partial breakdown is legitimate
- Verified in the database: NVDA Data Center FY2025 115,186 and FY2024 47,525; AAPL Greater China FY2025 64,377; COF Credit Card FY2024 28,164. Rows cite the 10-K they were read from (accession and filed date come from companyfacts, since the HTML does not carry them)
- Not exposed to the agent yet: get_facts defaults to consolidated rows, so get_financials is unchanged. Next PR: a `breakdown` argument on get_financials (returns every member on the axis, so the model never guesses names)

# 10-2-2026 - Segments follow-ups: stale rows, and a derived corporate remainder
- Bug found while explaining the NVDA FY2026 problem: after the corporate-equals-total rule, the database still held the bad 139,297 "Corporate and other" row. upsert_facts only inserts and updates, so a row the parser stops producing stays forever. ingest_segments now calls replace_segment_facts, which deletes the ticker's breakdown rows (axis != '') and inserts the new set in one transaction; consolidated rows are untouched. Verified: the 139,297 row is gone after re-ingesting
- Why operating income does not reconcile: companies keep costs out of their segments (unallocated stock compensation, R&D, headquarters) and report them on a reconciling line. NVIDIA FY2026 tags no such operating-income line; Apple's FY2020-22 filings tag the pieces under other concepts (R&D 26,251 as a reconciling item) but never the line itself. Revenue has nothing unallocated, so it adds up
- Chosen: when a business-segment breakdown has no reported corporate row, store "Corporate and other" = consolidated total - sum of segments, derived=True, citing the filing of the total. Same flag and idea as the derived Q4 (a subtraction of two company-reported figures). Restricted to the business-segment axis (product and geography members overlap, so the remainder would mean nothing) and to remainders above 1M
- Cost, stated plainly: a remainder absorbs whatever is left, so a parse miss on a segment would show up as a plausible corporate number. Mitigations: it is only made when no corporate figure was reported, it is flagged derived, it is printed at ingest, and the sum check ignores derived rows because they reconcile by construction. The check that protects against missed segments is the parser cross-check and the revenue breakdowns, which reconcile without any remainder (65 of 69 before; the 4 that did not are exactly the 4 remainders now derived)
- Derived values (checked by hand): NVDA FY2026 130,387 - 139,297 = -8,910; AAPL FY2022 119,437 - 152,895 = -33,458 (FY2020 -24,952, FY2021 -28,057)
- Deferred, wanted: the ASU 2023-07 filings also tag per-segment expenses (Apple cost of sales and operating expenses by region; Capital One provision for credit losses, non-interest expense, pre-tax income, net income by segment; NVIDIA depreciation by segment). The table already fits them (new metrics, same axis/segment columns); it needs more METRICS entries, hand-verified per company, and point-in-time values (loans, deposits) need a separate change because the parser reads only full-year durations

# 10-2-2026 - get_financials breakdown: segment, product and geography figures for the agent
- Interface: an optional `breakdown` argument ('product' | 'segment' | 'geography') on the existing get_financials. Omitted, nothing changes. Given, it returns every slice on that axis for the year, so the model never guesses member names; without fiscal_year it returns the latest two years so a growth question can take one call (chosen over one year: ~14 rows instead of ~7). Ids add axis and slice: NVDA_revenue_FY2025_product_DataCenter
- Guardrails: a breakdown a company does not have returns an error listing those it has, read from the database (COF by product). A quarter with a breakdown is an error ("annual only for now"), so the model cannot believe it filtered. A breakdown needs two real slices: Apple's R&D has a single "Corporate and other" row equal to the whole figure (R&D is unallocated), which is not a breakdown
- Overlap: slices can contain other slices (Data Center = Compute + Networking; Apple Products = iPhone + Mac + iPad + Wearables). find_parts marks a slice equal to the sum of two or more others as their parent (part_of on the children, plus a note not to add them). Detected from the numbers, chosen over a sentence in the description because prompt rules are followed loosely
- Bug found by calling the tool on real data: NVIDIA FY2025 revenue by geography summed to 154,181 against a 130,497 total. Its FY2026 10-K re-presented FY2025 geography on a new basis and dropped Singapore, and "latest filing wins" per slice kept the old Singapore row next to the new US/China/Taiwan rows. Now the latest filing wins per whole breakdown (metric, axis, year). Side effect, a cross-check: NVIDIA FY2024 and FY2025 corporate operating income now come out as derived remainders, -4,890 and -6,507, exactly the figures the FY2025 10-K reported
- The sum check had passed that broken year, because it accepted any subset of slices that matched the total (leaving out Singapore matched). It now drops the children find_parts detects and requires the remaining slices to add up. Result unchanged elsewhere: every breakdown reconciles except the six that get a derived remainder
- Live finding 1: "How much did NVIDIA's Data Center revenue grow in fiscal 2025?" was wrong in 3/3 runs: the model asked for breakdown=segment, got NVIDIA's reporting segment Compute and Networking (116,193 / 47,405) and called it Data Center (+145.1%). Fixes: the breakdown description says what each axis holds with examples, and each result lists the slice names on the company's other axes for that metric (data-driven, so it scales). After: 3/3 used product, 47,525 -> 115,186 = +142.37% via calculate, both ids cited
- Live finding 2: "Which Apple region had the most revenue in fiscal 2025?" answered "Other Countries" 199,994 in 3/3 runs, caused by my own description ("geography: countries or regions"). Apple's regions are its reporting segments. The description now says segments are regions for some companies and that geography's "Other Countries" is a remainder, not a region. After: 3/3 Americas 178,353. "Which Capital One segment earns the most revenue?" was right throughout (Credit Card, 39,560 FY2025)
- Regression, one run each, unchanged: NVDA revenue growth +114.2%, Apple FY2024 gross margin 46.21%, COF gross profit "does not report", Tesla refuses without searching, NVDA EPS 0.174 -> 1.193 (+585.63%, revised once by the citation check as before)
- Lesson: the wording of a tool's argument description steered the answer as much as the data did; both failures were the model faithfully following a description that was ambiguous or wrong

# 10-2-2026 - More segment metrics, and balances (values on a date)
- Probed which concepts the 12 cached 10-Ks tag per segment that METRICS did not read. Worth adding: cost of revenue (Apple by product FY2020-25 and by region FY2023-25), pre-tax income, provision for credit losses, non-interest expense (Capital One by segment FY2020-25), and Capital One loans and deposits by segment. NVIDIA gains almost nothing (depreciation only). Added six METRICS entries: cost_of_revenue, pretax_income, provision_for_credit_losses, noninterest_expense, loans, deposits. Each also gives the company-wide figure. Left out: Apple "operating expenses" by region (tagged SellingAndMarketingExpense, would be mislabelled) and Capital One net income by segment (tagged income from continuing operations, a different number)
- Every new series passes the checks added earlier: parser vs SEC API (NVDA 96, AAPL 96, COF 119 consolidated values, 0 mismatches), and all 48 Capital One breakdowns reconcile with no derived remainder. By hand: COF FY2024 pre-tax by segment 4,316 + 1,911 + 1,582 - 1,899 = 5,910 (the consolidated figure); Apple FY2025 cost of sales 194,116 + 26,844 = 220,960; COF 2025 loans 279,570 + 84,790 + 89,262 + 0 = 453,622 and deposits 0 + 423,932 + 31,250 + 20,589 = 475,771, both equal to the company-wide balances
- Balances: loans and deposits are a value on a date, not an amount over a period (spec balance=True). A row is the balance on the period's last day (period_start = period_end), labelled FY for the fiscal year end and Q1-Q3 for a 10-Q's quarter end, latest filing wins, and nothing is derived (no Q4 or Q2: balances are not subtracted). The parser reads them only at a date that is also a full-year end in the same filing, so mid-year dates in notes cannot be mistaken for year-end figures. The tool marks them balance=true with as_of, the description says never to add them across dates, and period=Q4 returns an error pointing to FY
- Scaling worry (new tickers): bottom-line concepts are the same across companies (net income, pre-tax income); top-line ones vary (revenue, cost of revenue have several names), industries differ (banks have no cost of revenue), and a concept can exist and be the wrong number (COF's fee-only revenue). So the per-company METRICS map will not scale by hand. Planned next PR, agent/metric-resolver: candidate concepts per metric, picked per company by accounting identities (revenue >= its parts, gross profit = revenue - cost, pre-tax - tax = net income) with a small printed override table, accepted only if it reproduces today's hand-verified map for NVDA, AAPL and COF
- Live finding (Apple Services gross margin, 0/3 correct twice): the model called company-wide gross_profit and revenue and reported 46.9% as the Services margin. First fix, a note listing a company's available breakdowns on plain results, did nothing in 3/3 runs (the model commits to the company-wide metric at step 1, before it sees the note). Second fix, a sentence in the tool description saying a question about one product line, segment or region must set breakdown, and that a margin for it needs revenue and cost of revenue with the same breakdown: 3/3 correct, 75.41% = (109,158 - 26,844) / 109,158. One of the three runs spent the whole 6-step budget on three calculate calls and was forced to answer (correct); another was revised once and ended with one uncited sentence. Kept the note: it costs ~180 chars per result and costs nothing when there are no breakdowns, and it is data-driven; but it is not what fixed the answer
- Also 3/3: Capital One highest pre-tax income 2024 (Credit Card, 4,316); most deposits at end of 2025 (Consumer Banking, 423,932). Regression, one run each, unchanged: Data Center +142.37%, Apple top region Americas, NVDA revenue +114.2%, Apple gross margin 46.21%, COF gross profit "does not report", Tesla refuses, NVDA EPS +585.63%; COF total deposits 475,771

# 10-2-2026 - Metric resolver: concepts chosen per company from its own data, not a hand-kept map
- Problem: METRICS mapped each metric to a concept per company, all hand-checked, so a new ticker meant reading its filings by hand. Surveyed 15 companies (the 3 in the corpus plus MSFT, WMT, JPM, BAC, TSLA, GOOGL, KO, AMZN, UNH, JNJ, CAT, XOM) before designing
- What the survey showed: net income, operating income, EPS, operating cash flow use the same concept everywhere. Revenue, cost of revenue, R&D and provision have 2-3 real synonyms, and when several exist the largest is the total (COF Revenues 53,434M vs a fee-only 8,062M; WMT 713,163M vs 706,413M without membership fees; CAT cost 44,752M vs a stray 49M; JNJ R&D 14,665M vs 109M). Two traps: look-alike concepts that are a different number ("pre-tax income, domestic" is only the US part; "interest and dividend income" is gross), and a synonym that means something else for that business (UNH cost of goods sold is only its pharmacy costs)
- Design (resolver.py, pure): METRICS now holds a synonym list per metric, written once for all companies and only true synonyms (look-alikes stay out). Each period takes its largest synonym, so a company that switches concept keeps every year; rows record the concept used. Synonyms that disagree in the latest year are accepted only if the largest satisfies an accounting identity (revenue - costs and expenses = operating income; revenue - cost - opex or SG&A = operating income; revenue - cost = gross profit; net interest + non-interest income = revenue), otherwise the metric is not stored and the reason is printed, so the agent says "not reported". Bank-only metrics resolve only for a company that passes the bank test (NII + non-interest income = revenue), because Caterpillar also tags a loan-loss provision (109M on 67,589M revenue). OVERRIDES pins what no rule settles (COF loans and deposits) and is printed at ingest. A missing core metric (revenue, net income, EPS, operating cash flow) fails the ticker's ingestion and prints concepts that fit the identity X - Y = operating income or gross profit as suggestions, never applied automatically
- Acceptance: on the cached SEC data the resolver picks exactly the hand-verified concepts for NVDA, AAPL and COF, and normalize() output matches the old map row for row, with 2 exceptions: NVDA FY2020 Q4 revenue and cost of revenue, same values (3,105M and 1,090M), now the company's own reported Q4 (its FY2020 10-K tagged it under a second synonym) instead of derived. Ingest row counts and every segment reconciliation are unchanged. The comparison runs in CI only when data/xbrl exists (it is git-ignored), so CI skips it
- Dry run over the 14 companies with data: all 4 core metrics resolve for every one; Exxon's file has 4 rows and no 10-K (wrong SEC entity for the ticker), reported as "no annual data" instead of crashing. SKIPPED, as designed: CAT cost of revenue and JNJ R&D (synonyms disagree, no identity). The coverage report also showed pre-tax income missing for CAT and AMZN (they tag the variant ...MinorityInterestAndIncomeLossFromEquityMethodInvestments, which can differ, so it is not listed as a synonym), and loans and deposits unresolved for BAC and JPM (no override). Report-only DIFFERS lines (pre-tax - tax vs net income, 1%) for COF, TSLA, UNH, WMT are minority interests and discontinued operations
- KNOWN LIMITATION, not fixed: UNH cost_of_revenue is stored (single synonym, 50,655M on 447,567M revenue) but it is only pharmacy costs; no identity can evaluate it (no gross profit, no opex line). A "gross margin" computed from it would mislead. The next PR (calculation linkbase) is where this would be caught
- Tool changes needed because the per-company table is gone: which metrics a company reports now comes from the facts table (store.available_metrics) and the tool description no longer lists them per company, saying instead that an unreported pair returns an error listing what the company has
- Live regression found a bug that already existed on main: "What was Apple's gross margin in fiscal 2024?" was wrong 4 of 8 runs there (-14.1% or -16.42%: the model fetched gross profit, then cost of revenue instead of revenue, and divided by the cost). It appeared when cost_of_revenue became a metric and was missed because one run passed. My removal of the per-company list made it worse, 0 of 8 correct on this branch. A sentence in the tool description only brought it back to 4 of 8; a note in the result of every profit metric ("A gross profit margin is this figure divided by revenue", flag profit=True in METRICS) gave 8 of 8 correct (46.21%)
- After: Apple Services margin 75.41% 3/3 (twice), NVDA Data Center +142.37% 3/3, COF gross profit "does not report" 3/3, NVDA net income +144.9% 3/3, one run each of Tesla refusal, EPS +585.63%, Apple top region Americas, COF deposits 475,771. Lesson: run a question several times before calling it a regression pass; the previous PR's single-run check hid a 50% failure

# 10-2-2026 - Calculation linkbase: the company's own income statement as a second opinion on the resolver
- Survey first (latest 10-K of 14 companies, fetched read-only): every one has a calculation linkbase (13 as a separate _cal.xml, Microsoft embedded in its .xsd schema). The income statement is found by role name (Income, Operations, Earnings, Results of Operations, INCOMESTATEMENTS; excluding comprehensive income and detail notes). Revenue is found by position in 13 of 14: the + line under gross profit, else under operating income, else (a bank) under pre-tax income. Walmart's tree says Revenues = RevenueFromContract + OtherIncome (membership fees), which is why "largest synonym wins" gives the total
- What the survey did not support: making the tree the primary method. Capital One's tree is fragmented (pre-tax income has no children; total revenue is not in the file), so the resolver has to stay. Position alone also does not catch UnitedHealth: its cost of goods sold sits inside "costs and expenses" exactly where Amazon's real cost of sales sits; only the numbers separate them (UnitedHealth's largest cost is medical claims, 313,995M, against 50,655M). I had said in the previous entry that the tree "would catch" UNH; it does not on its own. Company-specific concepts, the original reason to prefer the tree, mostly appear in sub-lines (Amazon fulfillment, JPMorgan fee lines) and only one touched a metric we track
- Chosen (user's decision): the tree as a second opinion and gap filler, one PR. statements.py fetches and caches the latest 10-K's linkbase (data/xbrl/<TICKER>_cal.xml), merges a role split over several blocks, finds the income statement and the position of each metric. resolve() uses it four ways: (1) report whether the statement confirms each choice; (2) when synonyms disagree and exactly one is a line of the statement, that one is the company's (Caterpillar cost of revenue 44,752M vs a stray 49M; J&J R&D 14,665M vs 109M, both previously skipped); (3) when no synonym matched, take the statement's concept if it is a standard concept with data (pre-tax income for Amazon and Caterpillar, which tag the ...MinorityInterestAndIncomeLossFromEquityMethodInvestments variant); (4) cost of revenue that is not the - line under gross profit must be the largest line of its cost group, otherwise it is not stored. Everything is attached as facts['income_statement'] by ingest and works without it: a missing or failing download only prints a line
- Cost-of-revenue rule, stated plainly: it rejects UnitedHealth (correct), keeps Amazon, Alphabet, Walmart and Caterpillar, and would also reject a company whose R&D exceeds its cost of revenue and which has no gross profit line (Meta). That is a false negative: the metric is not stored, never a wrong value, which matches the earlier choice of precision over recall
- Staleness check (works with or without the tree): a metric whose last 10-K year is 2 or more fiscal years behind the company's latest is not stored (Amazon gross profit stops at FY2009, J&J operating income at FY2014, BAC provision at FY2021), because the agent would present the old figure as the latest. One year behind is only flagged. BAC moved its provision to a company concept (bac:FinancingReceivable...CreditLossProvisionReversal) in 2022, so the resolver as merged had stored it only through FY2021 while the coverage report said "found"; the tree names the new concept, and the report says its values would need the 10-K HTML, which this PR does not read
- Existing bug found and fixed on the way: the latest fiscal year was taken from any filing, and Amazon's 10-Qs carry trailing-twelve-month figures that look like a year ending a quarter later (FY2026), which would have made every Amazon metric look stale. It now uses 10-K facts only
- Dry run, 14 companies, resolver alone vs with the tree: AAPL, COF, GOOGL, JPM, KO, MSFT, NVDA, TSLA, WMT unchanged and confirmed (JPM and TSLA name revenue differently from the resolver but carry the same figure, shown as agreement); CAT cost of revenue and pre-tax income, AMZN pre-tax income, JNJ R&D filled or settled; UNH cost of revenue rejected; BAC provision and AMZN and JNJ ancient metrics dropped as discontinued. For NVDA, AAPL and COF nothing changes: row counts, segment reconciliations and the agent's answers are identical, and the statement confirms every choice (COF's tree confirms pre-tax income and provision only, since it has no revenue position)
- Not done: reading values for company-specific concepts from the 10-K HTML (BAC provision would need it), and foreign filers (20-F). Both wait for a company that needs them
