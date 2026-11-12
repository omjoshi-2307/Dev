"""Shared contracts for DEV AI providers."""

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence


MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single message passed to an AI provider."""

    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Normalized text response returned by an AI provider."""

    content: str
    provider: str
    model: str


class ProviderConfigurationError(RuntimeError):
    """Raised when a provider is not configured sufficiently to run."""


class AIProvider(Protocol):
    """Contract implemented by all DEV text-generation providers."""

    name: str

    async def generate(
        self,
        messages: Sequence[ChatMessage],
    ) -> ProviderResponse:
        """Generate a response for the supplied conversation messages."""
