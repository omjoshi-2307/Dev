"""Centralized application settings for DEV."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Settings loaded from environment variables and the repository .env file."""

    app_name: str = "DEV"
    environment: str = "development"

    openai_api_key: str | None = None
    openai_model: str | None = None

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="DEV_",
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
