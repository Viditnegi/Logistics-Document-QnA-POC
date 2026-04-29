from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Logistics Document QA POC"
    upload_dir: Path = Field(default=Path("data/uploads"), validation_alias="UPLOAD_DIR")
    chroma_dir: Path = Field(default=Path("data/chroma"), validation_alias="CHROMA_DIR")
    chroma_collection: str = "logistics_documents"
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL")
    chat_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_CHAT_MODEL")
    retrieval_k: int = 6
    chunk_size: int = 1400
    chunk_overlap: int = 250
    confidence_threshold: float = 0.35


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def ensure_directories() -> None:
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
