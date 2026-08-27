from openai import OpenAI
from earnings_rag.config import settings

_llm_instance = None

def _llm() -> OpenAI:
    global _llm_instance
    if _llm_instance is None:
        if not settings.llm_api_key:
            raise RuntimeError('LLM_API_KEY not set')
        _llm_instance = OpenAI(api_key=settings.llm_api_key)
    return _llm_instance

PROMPT = """You are a financial research assistant. Answer the question using ONLY the context below, which comes from SEC 10-k filings.

Rules:
- If the context does not contain the answer, say "The provided filings do not address this."
- Cite sources by their [number] inline.
- Do not use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""