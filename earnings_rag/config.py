from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_configs = SettingsConfigDict(env_file='.env', extra='ignore')

    db_host: str = 'localhost'
    db_port: int = 5432
    db_