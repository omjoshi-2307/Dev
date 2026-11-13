"""Centralized application settings for DEV."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.app.router.enums import PrivacyLevel, ProviderName

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

    # Router configuration defaults
    router_default_privacy_level: PrivacyLevel = Field(
        default=PrivacyLevel.LOCAL_ONLY,
        validation_alias="DEV_ROUTER_DEFAULT_PRIVACY_LEVEL",
        description="Mandatory baseline privacy level. Must remain 'local_only' to prevent implicit cloud processing.",
    )
    router_default_provider: ProviderName = Field(
        default=ProviderName.AUTO,
        validation_alias="DEV_ROUTER_DEFAULT_PROVIDER",
        description="Default provider preference when not explicitly specified in a request.",
    )
    router_allow_fallback: bool = Field(
        default=False,
        validation_alias="DEV_ROUTER_ALLOW_FALLBACK",
        description="Default fallback policy when not explicitly specified in a request.",
    )

    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("router_default_privacy_level", mode="before")
    @classmethod
    def validate_privacy_default(cls, value: object) -> PrivacyLevel:
        """Enforce that default privacy is strictly local_only and cannot enable cloud processing implicitly."""
        if isinstance(value, str):
            try:
                level = PrivacyLevel(value.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid privacy level '{value}'. Must be 'local_only'."
                )
        elif isinstance(value, PrivacyLevel):
            level = value
        else:
            raise ValueError(f"Invalid privacy level type: {type(value)}")

        if level != PrivacyLevel.LOCAL_ONLY:
            raise ValueError(
                "Default privacy level must be 'local_only'. "
                "Cloud processing cannot be enabled implicitly by environment configuration; "
                "clients must explicitly request 'cloud_allowed' per request."
            )
        return level

    @field_validator("router_default_provider", mode="before")
    @classmethod
    def validate_provider_default(cls, value: object) -> ProviderName:
        """Validate that router_default_provider is a recognized ProviderName."""
        if isinstance(value, str):
            try:
                return ProviderName(value.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid provider '{value}'. Must be one of {[p.value for p in ProviderName]}."
                )
        elif isinstance(value, ProviderName):
            return value
        raise ValueError(f"Invalid provider type: {type(value)}")


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
