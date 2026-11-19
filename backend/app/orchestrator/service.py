"""Chat orchestrator connecting deterministic routing with AI providers."""

from __future__ import annotations

import ipaddress
import logging
import urllib.parse
from collections.abc import Mapping, Sequence
from typing import Any

from backend.app.config import Settings, get_settings
from backend.app.providers.base import (
    AIProvider,
    ChatMessage,
    ProviderResponse,
)
from backend.app.providers.ollama_provider import (
    OllamaProvider,
    validate_injected_client_for_local_endpoint,
)
from backend.app.providers.openai_provider import OpenAIProvider
from backend.app.router.enums import PrivacyLevel, ProviderName
from backend.app.router.policy import RoutingPolicyError, route_request
from backend.app.router.schemas import (
    RESERVED_METADATA_KEYS,
    ChatMessageInput,
    ChatRequest,
    ChatResponse,
    ConcreteProvider,
    RoutingDecision,
)

logger = logging.getLogger(__name__)


def validate_local_destination(url: str | None) -> None:
    """Validate that a destination endpoint is a verified local loopback address.

    Enforces that LOCAL_ONLY requests can only execute against a verified local
    endpoint. Rejects remote hostnames, external IPs, ambiguous formats, and URLs
    with userinfo, failing closed before transmitting any message content.

    Accepts:
    - Schemes: http or https
    - Hostnames: exact 'localhost' or 'localhost.'
    - IP literals: IPv4 loopback (127.0.0.0/8) or IPv6 loopback (::1)

    Rejects:
    - Missing or non-string URLs
    - Non-http/https schemes
    - Remote hostnames and domain names (even if containing 'localhost' as a substring)
    - Non-loopback IPs (e.g. LAN, private, public)
    - URLs containing userinfo (user:pass@)
    - Ambiguous, malformed, or unparseable URLs
    """
    if not isinstance(url, str) or not url.strip():
        raise RoutingPolicyError(
            "LOCAL_ONLY destination boundary violation: local provider endpoint is not configured."
        )

    try:
        parsed = urllib.parse.urlsplit(url.strip())
    except Exception as exc:
        raise RoutingPolicyError(
            f"LOCAL_ONLY destination boundary violation: failed to parse endpoint URL: {exc}"
        ) from exc

    if parsed.scheme not in ("http", "https"):
        raise RoutingPolicyError(
            f"LOCAL_ONLY destination boundary violation: unsupported scheme '{parsed.scheme}'. "
            "Only 'http' and 'https' are permitted."
        )

    if parsed.username or parsed.password:
        raise RoutingPolicyError(
            "LOCAL_ONLY destination boundary violation: userinfo in endpoint URL is not permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise RoutingPolicyError(
            "LOCAL_ONLY destination boundary violation: endpoint URL missing valid hostname."
        )

    hostname = hostname.lower()

    # Exact loopback hostnames (RFC 6761)
    if hostname in ("localhost", "localhost."):
        return

    # Check for IP literal loopback (IPv4 127.0.0.0/8 and IPv6 ::1)
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_loopback:
            return
        raise RoutingPolicyError(
            f"LOCAL_ONLY destination boundary violation: IP address '{hostname}' is not a local loopback address."
        )
    except ValueError:
        pass

    # Reject any non-loopback hostname
    raise RoutingPolicyError(
        f"LOCAL_ONLY destination boundary violation: host '{hostname}' is not a verified local loopback destination."
    )


def validate_provider_identity(
    expected_key: ConcreteProvider | ProviderName,
    provider: Any,
) -> None:
    """Validate that an injected or resolved provider matches its registered routing key.

    Prevents dependency-injection bypasses such as executing an OpenAIProvider
    under the OLLAMA key, or injecting providers with missing, malformed, or
    mismatched identity information.
    """
    name = getattr(provider, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise RoutingPolicyError(
            "Provider identity unverifiable: provider is missing a valid 'name' attribute."
        )

    norm_name = name.strip().lower()

    if expected_key == ProviderName.OLLAMA:
        if isinstance(provider, OpenAIProvider):
            raise RoutingPolicyError(
                "Provider identity mismatch: OpenAIProvider cannot be registered or "
                "executed under provider key 'ollama'."
            )
        if norm_name != "ollama":
            raise RoutingPolicyError(
                f"Provider identity mismatch: provider with name '{name}' cannot be "
                "registered or executed under provider key 'ollama'."
            )
    elif expected_key == ProviderName.OPENAI:
        if isinstance(provider, OllamaProvider):
            raise RoutingPolicyError(
                "Provider identity mismatch: OllamaProvider cannot be registered or "
                "executed under provider key 'openai'."
            )
        if norm_name != "openai":
            raise RoutingPolicyError(
                f"Provider identity mismatch: provider with name '{name}' cannot be "
                "registered or executed under provider key 'openai'."
            )


def validate_provider_model(
    provider_name: ConcreteProvider | ProviderName,
    provider: Any,
    expected_model: str,
) -> None:
    """Ensure a built-in provider adapter executes the model selected by the router.

    Injected adapters may carry settings that differ from the orchestrator's settings.
    Reject that mismatch before sending a request so the response cannot claim a model
    different from the one the adapter actually executes.
    """
    configured_model: str | None = None

    if provider_name == ProviderName.OLLAMA and type(provider) is OllamaProvider:
        provider_settings = getattr(provider, "_settings", None)
        configured_model = getattr(provider_settings, "ollama_model", None)
    elif provider_name == ProviderName.OPENAI and type(provider) is OpenAIProvider:
        provider_settings = getattr(provider, "_settings", None)
        configured_model = getattr(provider_settings, "openai_model", None)
    else:
        # Test doubles and third-party adapters expose no standard settings contract.
        return

    if configured_model != expected_model:
        raise RoutingPolicyError(
            f"Provider model configuration mismatch: router selected '{expected_model}' "
            f"for '{provider_name.value}', but the adapter is configured for '{configured_model}'."
        )


def validate_trusted_local_ollama_provider(provider: Any) -> None:
    """Validate that provider is the exact, unmodified, trusted OllamaProvider implementation.

    Enforces that LOCAL_ONLY requests can only execute through the genuine,
    un-subclassed, and un-overridden OllamaProvider adapter, preventing privacy
    boundary bypasses via subclassing, custom classes, or instance-level method monkeypatching.
    """
    if type(provider) is not OllamaProvider:
        prov_cls_name = getattr(provider, "__class__", type(provider)).__name__
        raise RoutingPolicyError(
            f"Privacy boundary violation: LOCAL_ONLY requests require the exact trusted "
            f"OllamaProvider adapter; custom or subclassed provider '{prov_cls_name}' is rejected."
        )

    # Reject instance-level method monkeypatching or overriding
    if "generate" in getattr(provider, "__dict__", {}):
        raise RoutingPolicyError(
            "Privacy boundary violation: LOCAL_ONLY requests reject OllamaProvider instances "
            "with overridden 'generate' methods."
        )

    # Verify that the effective generate method is the original OllamaProvider.generate implementation
    gen_method = getattr(provider, "generate", None)
    if not callable(gen_method):
        raise RoutingPolicyError(
            "Privacy boundary violation: OllamaProvider missing callable 'generate' method."
        )

    if getattr(gen_method, "__func__", None) is not OllamaProvider.generate:
        raise RoutingPolicyError(
            "Privacy boundary violation: OllamaProvider 'generate' method has been modified or overridden."
        )


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
                validate_provider_identity(norm_key, provider)
                self._providers[norm_key] = provider

        if ollama_provider is not None:
            validate_provider_identity(ProviderName.OLLAMA, ollama_provider)
            self._providers[ProviderName.OLLAMA] = ollama_provider

        if openai_provider is not None:
            validate_provider_identity(ProviderName.OPENAI, openai_provider)
            self._providers[ProviderName.OPENAI] = openai_provider

    def _resolve_provider_endpoint(
        self,
        provider_name: ConcreteProvider | ProviderName,
        *,
        provider: AIProvider | None = None,
    ) -> str | None:
        """Resolve the configured target endpoint URL for a provider.

        For injected providers, resolves the endpoint directly from the provider's
        own configuration (base_url). Never infers or defaults an injected provider's
        target destination from unrelated orchestrator settings.
        """
        if provider_name == ProviderName.OLLAMA:
            target_prov = provider or self._providers.get(ProviderName.OLLAMA)
            if target_prov is not None:
                if hasattr(target_prov, "base_url") and isinstance(target_prov.base_url, str) and target_prov.base_url.strip():
                    return target_prov.base_url.strip()
                # An injected provider without a verified base_url cannot be verified
                return None
            return getattr(self._settings, "ollama_base_url", None)
        return None

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
        3. LOCAL_ONLY requests only execute against verified local loopback endpoints.
        4. Only the selected provider is invoked; failures never trigger runtime fallback.
        5. Response provider and model strictly reflect the authoritative routing decision.
        """
        # 1. Authoritative deterministic routing
        decision: RoutingDecision = route_request(request, settings=self._settings)

        # 2. Defense-in-depth privacy boundary and provider identity enforcement
        provider = self._get_provider(decision.selected_provider)
        validate_provider_identity(decision.selected_provider, provider)
        validate_provider_model(decision.selected_provider, provider, decision.model)

        if request.privacy_level == PrivacyLevel.LOCAL_ONLY:
            if decision.selected_provider != ProviderName.OLLAMA:
                raise RoutingPolicyError(
                    f"Privacy boundary violation: request with privacy_level='local_only' "
                    f"cannot execute provider '{decision.selected_provider}'."
                )

            # Ensure provider is strictly the exact trusted OllamaProvider implementation.
            # Subclasses and custom implementations overriding generate() are rejected.
            validate_trusted_local_ollama_provider(provider)

            # Enforce that LOCAL_ONLY destination is strictly a verified local loopback endpoint
            # Resolved directly from the provider itself (never inferred from unrelated orchestrator settings)
            endpoint = self._resolve_provider_endpoint(decision.selected_provider, provider=provider)
            validate_local_destination(endpoint)

            # Enforce that injected provider transports do not have proxy mounts or active proxy leakage
            if hasattr(provider, "_http_client") and provider._http_client is not None:
                validate_injected_client_for_local_endpoint(provider._http_client)

        # 3. Message conversion
        provider_messages = convert_messages(request.messages)

        # 4. Provider execution
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
        # Filter any reserved system/diagnostic keys from client metadata as defense-in-depth
        response_metadata: dict[str, Any] = {
            k: v
            for k, v in request.metadata.items()
            if str(k).lower() not in RESERVED_METADATA_KEYS
        }

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
