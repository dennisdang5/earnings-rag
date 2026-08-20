from openai import OpenAI
from earnings_rag.config import settings

_client_instance = None
def _client() -> OpenAI:
    global _client_instance
    if _client_instance is None:
        if not settings.embedding_api_key:
            raise RuntimeError('EMBEDDING_API_KEY not set')
        _client_instance = OpenAI(api_key=settings.embedding_api_key)
    return _client_instance



