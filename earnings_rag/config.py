from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    db_host: str = 'localhost'
    db_port: int = 5432
    db_name: str = 'earnings'
    db_user: str = 'postgres'
    db_password: str = 'postgres'

    sec_user_agent: str = ''
    embedding_api_key: str = ''
    llm_api_key: str = ''

    # Embedding model and dimension must match thus the model determines the dimension
    embedding_model: str = 'text-embedding-3-small'
    embedding_dim: int = 1536
    embedding_batch_size: int = 100

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    top_k: int = 5

    data_dir: Path = REPO_ROOT / 'data'

    tickers: list[str] = ['NVDA', 'AAPL', 'COF']
    filings_per_ticker: int = 4
    form_type: str = '10-K'

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / 'raw'

    @property
    def db_url(self) -> str:
        return (
            f'postgresql://{self.db_user}:{self.db_password}'
            f'@{self.db_host}:{self.db_port}/{self.db_name}'
        )

settings = Settings()