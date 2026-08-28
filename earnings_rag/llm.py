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
- Every claim must be supported by a specific numbered source. If a claim is not directly stated in the context, omit it.
- Cite only sources you actually used. Do not cite a source that does not contain the claim.
- If the context is only tangentially related, say the filings do not clearly address the question.

Context:
{context}

Question: {question}

Answer:"""

def build_context(hits: list[dict]) -> str:
    """
    :param hits: Data returned from retrieve() such that it a list of dictionary that contain id, timestamp, ticker, text, and distance from our query
    :return: Labeled string that was converted from list of dictionary so the model can read
    """
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f'[{i}] {hit['ticker']} {hit['period']}\n{hit['text']}')
    return '\n\n'.join(blocks)

def generate(question: str, hits: list[dict]) -> str:
    prompt = PROMPT.format(context=build_context(hits), question=question)

    response = _llm().chat.completions.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=0, # Controls randomness in token selection, 0 means we take the most likely token everytime
        messages=[{'role': 'user', 'content':prompt}],
    )

    return response.choices[0].message.content # API can return multiple completions per request and we just take first index