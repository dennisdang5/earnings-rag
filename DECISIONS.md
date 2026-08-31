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
- 