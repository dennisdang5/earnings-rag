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
