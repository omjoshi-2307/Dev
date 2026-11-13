"""Enums for the DEV AI router."""

from enum import Enum


class PrivacyLevel(str, Enum):
    """Privacy boundaries for routing AI requests."""

    LOCAL_ONLY = "local_only"
    CLOUD_ALLOWED = "cloud_allowed"


class TaskComplexity(str, Enum):
    """Complexity classification of a task."""

    AUTO = "auto"
    SIMPLE = "simple"
    COMPLEX = "complex"


class ProviderName(str, Enum):
    """Supported provider names and auto-selection option."""

    AUTO = "auto"
    OLLAMA = "ollama"
    OPENAI = "openai"
