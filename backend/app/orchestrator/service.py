"""Chat orchestrator connecting deterministic routing with AI providers."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from backend.app.config import Settings, get_settings
from backend.app.providers.base import (
    AIProvider,
    ChatMessage,
    ProviderResponse,
)
from backend.app.providers.ollama_provider import OllamaProvider
from backend.app.providers.openai_provider import OpenAIProvider
from backend.app.router.enums import PrivacyLevel, ProviderName
from backend.app.router.policy import RoutingPolicyError, route_request
from backend.app.router.schemas import (
    ChatMessageInput,
    ChatRequest,
    ChatResponse,
    ConcreteProvider,
    RoutingDecision,
)

logger = logging.getLogger(__name__)


def convert_messages(messages: Sequence[ChatMessageInput]) -> list[ChatMessage]:
    """Convert validated client message inputs into provider-level chat messages.

    Preserves message ordering, roles, and content. Rejects client-supplied system
    messages to ensure DEV maintains authoritative control over system instructions.
    """
    if not messages:
        raise ValueError("At least one message is required.")

    converted: list[ChatMessage] = []
    for msg in messages:
        if msg.role not in ("user", "assistant"):
            raise ValueError(
                f"Client message role '{msg.role}' is not permitted. "
                "Only 'user' and 'assistant' roles are allowed."
            )
        converted.append(ChatMessage(role=msg.role, content=msg.content))
    return converted


class ChatOrchestrator:
    """Orchestrates AI chat requests using deterministic routing and provider adapters."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        providers: Mapping[ConcreteProvider | ProviderName | str, AIProvider] | None = None,
        ollama_provider: AIProvider | None = None,
        openai_provider: AIProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._providers: dict[ProviderName, AIProvider] = {}

        if providers:
            for key, provider in providers.items():
                norm_key = ProviderName(key) if isinstance(key, str) else key
                self._providers[norm_key] = provider

        if ollama_provider is not None:
            self._providers[ProviderName.OLLAMA] = ollama_provider

        if openai_provider is not None:
            self._providers[ProviderName.OPENAI] = openai_provider

    def _get_provider(self, provider_name: ConcreteProvider | ProviderName) -> AIProvider:
        """Resolve or lazily instantiate the requested provider adapter."""
        if provider_name in self._providers:
            return self._providers[provider_name]

        if provider_name == ProviderName.OLLAMA:
            provider = OllamaProvider(settings=self._settings)
            self._providers[ProviderName.OLLAMA] = provider
            return provider

        if provider_name == ProviderName.OPENAI:
            provider = OpenAIProvider(settings=self._settings)
            self._providers[ProviderName.OPENAI] = provider
            return provider

        raise RoutingPolicyError(f"Unsupported provider '{provider_name}'.")

    async def handle_chat_request(self, request: ChatRequest) -> ChatResponse:
        """Execute a validated chat request through routing and provider adapters.

        Privacy and execution guarantees:
        1. Routing is purely deterministic via route_request().
        2. LOCAL_ONLY requests never execute cloud providers (OpenAI).
        3. Only the selected provider is invoked; failures never trigger runtime fallback.
        4. Response provider and model strictly reflect the authoritative routing decision.
        """
        # 1. Authoritative deterministic routing
        decision: RoutingDecision = route_request(request, settings=self._settings)

        # 2. Defense-in-depth privacy boundary enforcement
        if (
            request.privacy_level == PrivacyLevel.LOCAL_ONLY
            and decision.selected_provider != ProviderName.OLLAMA
        ):
            raise RoutingPolicyError(
                f"Privacy boundary violation: request with privacy_level='local_only' "
                f"cannot execute provider '{decision.selected_provider}'."
            )

        # 3. Message conversion
        provider_messages = convert_messages(request.messages)

        # 4. Provider resolution & execution
        provider = self._get_provider(decision.selected_provider)
        provider_response = await provider.generate(provider_messages)

        # 5. Output validation
        if not isinstance(provider_response, ProviderResponse):
            raise ValueError(
                "Provider returned an invalid response object."
            )

        if not isinstance(provider_response.content, str) or not provider_response.content.strip():
            raise ValueError(
                "Provider returned empty or whitespace-only content."
            )

        # 6. Response normalization & audit preservation
        response_metadata: dict[str, Any] = dict(request.metadata)

        # Detect provider identity mismatches safely without trusting provider self-reporting
        if provider_response.provider != decision.selected_provider:
            response_metadata["provider_reported_name"] = str(provider_response.provider)
            response_metadata["identity_mismatch"] = True

        if provider_response.model != decision.model:
            response_metadata["provider_reported_model"] = str(provider_response.model)
            response_metadata["model_mismatch"] = True

        return ChatResponse(
            content=provider_response.content,
            provider=decision.selected_provider,
            model=decision.model,
            routing=decision,
            metadata=response_metadata,
        )

    async def orchestrate(self, request: ChatRequest) -> ChatResponse:
        """Convenience alias for handle_chat_request."""
        return await self.handle_chat_request(request)
