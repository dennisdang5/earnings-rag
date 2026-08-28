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