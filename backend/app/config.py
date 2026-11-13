"""Centralized application settings for DEV."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Settings loaded from environment variables and the repository .env file."""

    app_name: str = Field(default="DEV", validation_alias="DEV_APP_NAME")
    environment: str = Field(
        default="development",
        validation_alias="DEV_ENVIRONMENT",
    )

    openai_api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    openai_model: str | None = Field(
        default=None,
        validation_alias="OPENAI_MODEL",
    )

    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_model: str | None = Field(
        default=None,
        validation_alias="OLLAMA_MODEL",
    )

    router_default_privacy_level: str = Field(
        default="local_only",
        validation_alias="DEV_ROUTER_DEFAULT_PRIVACY_LEVEL",
    )
    router_default_provider: str = Field(
        default="auto",
        validation_alias="DEV_ROUTER_DEFAULT_PROVIDER",
    )
    router_allow_fallback: bool = Field(
        default=False,
        validation_alias="DEV_ROUTER_ALLOW_FALLBACK",
    )

    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
