from pydantic_settings import BaseSettings, SettingsConfigDict

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

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    top_k: int = 5

    @property
    def db_url(self) -> str:
        return (
            f'postgresql://{self.db_user}:{self.db_password}'
            f'@{self.db_host}:{self.db_port}/{self.db_name}'
        )

settings = Settings()