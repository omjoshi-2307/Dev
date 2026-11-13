"""Unit tests for the DEV Chat Orchestrator."""

from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from backend.app.config import Settings
from backend.app.orchestrator.service import ChatOrchestrator, convert_messages
from backend.app.providers.base import (
    ChatMessage,
    ProviderConfigurationError,
    ProviderResponse,
)
from backend.app.providers.ollama_provider import OllamaProvider
from backend.app.router.enums import PrivacyLevel, ProviderName, TaskComplexity
from backend.app.router.policy import RoutingPolicyError
from backend.app.router.schemas import (
    ChatMessageInput,
    ChatRequest,
    ChatResponse,
    RoutingDecision,
)


class MockProvider:
    """Configurable mock AI provider for isolated orchestrator unit testing."""

    def __init__(
        self,
        name: str = "mock",
        response_content: str = "Default mock response text",
        response_provider: str | None = None,
        response_model: str = "mock-model",
        error_to_raise: Exception | None = None,
        raw_response: Any = None,
    ) -> None:
        self.name = name
        self.response_content = response_content
        self.response_provider = response_provider or name
        self.response_model = response_model
        self.error_to_raise = error_to_raise
        self.raw_response = raw_response
        self.recorded_messages: list[Sequence[ChatMessage]] = []
        self.call_count: int = 0

    async def generate(self, messages: Sequence[ChatMessage]) -> Any:
        self.call_count += 1
        self.recorded_messages.append(messages)
        if self.error_to_raise is not None:
            raise self.error_to_raise
        if self.raw_response is not None:
            return self.raw_response
        return ProviderResponse(
            content=self.response_content,
            provider=self.response_provider,
            model=self.response_model,
        )


def make_mock_ollama_provider(
    settings: Settings,
    response_content: str = "Local Ollama generated response.",
    response_model: str = "llama3.2:latest",
    error_to_raise: Exception | None = None,
    raw_response: Any = None,
    status_code: int = 200,
) -> tuple[OllamaProvider, list[dict[str, Any]]]:
    """Create a real OllamaProvider backed by a controlled in-memory MockTransport."""
    recorded_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        recorded_requests.append({"url": str(request.url), "json": body})

        if error_to_raise is not None:
            raise error_to_raise

        if raw_response is not None:
            return httpx.Response(status_code, json=raw_response, request=request)

        return httpx.Response(
            status_code,
            json={
                "message": {"role": "assistant", "content": response_content},
                "model": response_model,
            },
            request=request,
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    provider = OllamaProvider(settings=settings, http_client=client)
    return provider, recorded_requests


@pytest.fixture
def both_configured_settings() -> Settings:
    return Settings(
        OPENAI_API_KEY="test-openai-key",
        OPENAI_MODEL="gpt-4o-mini",
        OLLAMA_MODEL="llama3.2:latest",
    )


@pytest.fixture
def ollama_only_settings() -> Settings:
    return Settings(
        OPENAI_API_KEY=None,
        OPENAI_MODEL=None,
        OLLAMA_MODEL="llama3.2:latest",
    )


@pytest.fixture
def openai_only_settings() -> Settings:
    return Settings(
        OPENAI_API_KEY="test-openai-key",
        OPENAI_MODEL="gpt-4o-mini",
        OLLAMA_MODEL=None,
    )


@pytest.fixture
def none_configured_settings() -> Settings:
    return Settings(
        OPENAI_API_KEY=None,
        OPENAI_MODEL=None,
        OLLAMA_MODEL=None,
    )


# ---------------------------------------------------------------------------
# Requirement 1: Successful Ollama execution
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_successful_ollama_execution(both_configured_settings: Settings) -> None:
    ollama_provider, recorded = make_mock_ollama_provider(
        both_configured_settings,
        response_content="Local Ollama generated response.",
        response_model="llama3.2:latest",
    )
    openai_mock = MockProvider(name="openai")

    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_provider,
        openai_provider=openai_mock,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Hello local DEV!")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    response = await orchestrator.handle_chat_request(request)

    assert isinstance(response, ChatResponse)
    assert response.content == "Local Ollama generated response."
    assert response.provider == ProviderName.OLLAMA
    assert response.model == "llama3.2:latest"
    assert response.routing.selected_provider == ProviderName.OLLAMA
    assert len(recorded) == 1
    assert openai_mock.call_count == 0


# ---------------------------------------------------------------------------
# Requirement 2: Successful OpenAI execution
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_successful_openai_execution(both_configured_settings: Settings) -> None:
    ollama_provider, recorded = make_mock_ollama_provider(both_configured_settings)
    openai_mock = MockProvider(
        name="openai",
        response_content="OpenAI generated cloud response.",
        response_model="gpt-4o-mini",
    )

    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_provider,
        openai_provider=openai_mock,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Solve complex math proof.")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        complexity=TaskComplexity.COMPLEX,
    )

    response = await orchestrator.handle_chat_request(request)

    assert isinstance(response, ChatResponse)
    assert response.content == "OpenAI generated cloud response."
    assert response.provider == ProviderName.OPENAI
    assert response.model == "gpt-4o-mini"
    assert response.routing.selected_provider == ProviderName.OPENAI
    assert openai_mock.call_count == 1
    assert len(recorded) == 0


# ---------------------------------------------------------------------------
# Requirement 3: Correct message conversion and ordering
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_correct_message_conversion_and_ordering(
    both_configured_settings: Settings,
) -> None:
    ollama_provider, recorded = make_mock_ollama_provider(both_configured_settings)
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_provider,
    )

    input_messages = [
        ChatMessageInput(role="user", content="Step 1: First user message"),
        ChatMessageInput(role="assistant", content="Step 2: Assistant reply"),
        ChatMessageInput(role="user", content="Step 3: Follow-up question"),
    ]

    request = ChatRequest(
        messages=input_messages,
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    await orchestrator.handle_chat_request(request)

    assert len(recorded) == 1
    payload_messages = recorded[0]["json"]["messages"]
    assert len(payload_messages) == 3
    for orig, conv in zip(input_messages, payload_messages, strict=True):
        assert conv["role"] == orig.role
        assert conv["content"] == orig.content


def test_convert_messages_direct_helper() -> None:
    inputs = [
        ChatMessageInput(role="user", content="Hello"),
        ChatMessageInput(role="assistant", content="Hi"),
    ]
    converted = convert_messages(inputs)
    assert len(converted) == 2
    assert converted[0] == ChatMessage(role="user", content="Hello")
    assert converted[1] == ChatMessage(role="assistant", content="Hi")

    with pytest.raises(ValueError, match="At least one message is required"):
        convert_messages([])


# ---------------------------------------------------------------------------
# Requirement 4: Correct routing decision preservation
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_correct_routing_decision_preservation(
    openai_only_settings: Settings,
) -> None:
    # When preferred is Ollama, but only OpenAI is configured and fallback is allowed:
    ollama_mock = MockProvider(name="ollama")
    openai_mock = MockProvider(
        name="openai",
        response_content="Fallback OpenAI reply",
        response_model="gpt-4o-mini",
    )

    orchestrator = ChatOrchestrator(
        settings=openai_only_settings,
        ollama_provider=ollama_mock,
        openai_provider=openai_mock,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Please help")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        preferred_provider=ProviderName.OLLAMA,
        allow_fallback=True,
    )

    response = await orchestrator.handle_chat_request(request)

    assert response.provider == ProviderName.OPENAI
    assert response.routing.is_fallback is True
    assert response.routing.fallback_reason is not None
    assert "not configured" in response.routing.fallback_reason
    assert response.routing.selected_provider == ProviderName.OPENAI
    assert response.routing.evaluated_providers == [
        ProviderName.OLLAMA,
        ProviderName.OPENAI,
    ]


# ---------------------------------------------------------------------------
# Requirement 5: Correct normalized response fields and metadata
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_correct_normalized_response_fields(
    both_configured_settings: Settings,
) -> None:
    ollama_provider, recorded = make_mock_ollama_provider(
        both_configured_settings,
        response_content="Detailed normalized answer.",
        response_model="llama3.2:latest",
    )
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_provider,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Compute metrics.")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
        metadata={"client_session_id": "session-12345", "trace_id": "xyz-789"},
    )

    response = await orchestrator.handle_chat_request(request)

    assert response.content == "Detailed normalized answer."
    assert response.provider == ProviderName.OLLAMA
    assert response.model == "llama3.2:latest"
    assert isinstance(response.routing, RoutingDecision)
    assert response.metadata["client_session_id"] == "session-12345"
    assert response.metadata["trace_id"] == "xyz-789"
    assert len(recorded) == 1


# ---------------------------------------------------------------------------
# Requirement 6: Explicit provider selection
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_explicit_provider_selection_ollama(
    both_configured_settings: Settings,
) -> None:
    ollama_mock = MockProvider(name="ollama", response_content="Explicit Ollama")
    openai_mock = MockProvider(name="openai", response_content="Explicit OpenAI")

    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_mock,
        openai_provider=openai_mock,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Use local")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        preferred_provider=ProviderName.OLLAMA,
    )

    response = await orchestrator.handle_chat_request(request)
    assert response.provider == ProviderName.OLLAMA
    assert ollama_mock.call_count == 1
    assert openai_mock.call_count == 0


@pytest.mark.anyio
async def test_explicit_provider_selection_openai(
    both_configured_settings: Settings,
) -> None:
    ollama_mock = MockProvider(name="ollama", response_content="Explicit Ollama")
    openai_mock = MockProvider(name="openai", response_content="Explicit OpenAI")

    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_mock,
        openai_provider=openai_mock,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Use cloud")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        preferred_provider=ProviderName.OPENAI,
    )

    response = await orchestrator.handle_chat_request(request)
    assert response.provider == ProviderName.OPENAI
    assert openai_mock.call_count == 1
    assert ollama_mock.call_count == 0


# ---------------------------------------------------------------------------
# Requirement 7: LOCAL_ONLY never invokes OpenAI
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_local_only_never_invokes_openai(
    both_configured_settings: Settings,
) -> None:
    ollama_provider, recorded = make_mock_ollama_provider(
        both_configured_settings, response_content="Local reply"
    )
    openai_mock = MockProvider(name="openai", response_content="Cloud reply")

    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_provider,
        openai_provider=openai_mock,
    )

    # 1. Complex task under LOCAL_ONLY must route to Ollama, never OpenAI
    request_complex = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Complex task")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
        complexity=TaskComplexity.COMPLEX,
    )
    resp = await orchestrator.handle_chat_request(request_complex)
    assert resp.provider == ProviderName.OLLAMA
    assert openai_mock.call_count == 0
    assert len(recorded) == 1

    # 2. Explicit request for OpenAI under LOCAL_ONLY must be rejected
    request_explicit_cloud = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Prohibited request")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
        preferred_provider=ProviderName.OPENAI,
    )
    with pytest.raises(RoutingPolicyError, match="violates privacy level 'local_only'"):
        await orchestrator.handle_chat_request(request_explicit_cloud)
    assert openai_mock.call_count == 0


@pytest.mark.anyio
async def test_local_only_defense_in_depth_check() -> None:
    # Test that even if route_request were mocked to select OpenAI under LOCAL_ONLY,
    # the orchestrator's defense-in-depth rejects it before calling the provider.
    openai_mock = MockProvider(name="openai")
    orchestrator = ChatOrchestrator(openai_provider=openai_mock)

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Test")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    # Monkeypatch route_request to return an illegal decision
    from unittest.mock import patch

    illegal_decision = RoutingDecision(
        selected_provider=ProviderName.OPENAI,
        model="gpt-4o-mini",
        reason="Illegal decision",
    )
    with patch(
        "backend.app.orchestrator.service.route_request",
        return_value=illegal_decision,
    ):
        with pytest.raises(RoutingPolicyError, match="Privacy boundary violation"):
            await orchestrator.handle_chat_request(request)

    assert openai_mock.call_count == 0


# ---------------------------------------------------------------------------
# Requirement 8: Routing policy failure prevents provider execution
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_routing_policy_failure_prevents_provider_execution(
    none_configured_settings: Settings,
) -> None:
    ollama_mock = MockProvider(name="ollama")
    openai_mock = MockProvider(name="openai")

    orchestrator = ChatOrchestrator(
        settings=none_configured_settings,
        ollama_provider=ollama_mock,
        openai_provider=openai_mock,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Will fail routing")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    with pytest.raises(RoutingPolicyError):
        await orchestrator.handle_chat_request(request)

    # Neither provider was ever executed
    assert ollama_mock.call_count == 0
    assert openai_mock.call_count == 0


# ---------------------------------------------------------------------------
# Requirement 9: Provider execution failure does not invoke another provider
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_ollama_runtime_failure_does_not_invoke_openai(
    both_configured_settings: Settings,
) -> None:
    ollama_provider, recorded = make_mock_ollama_provider(
        both_configured_settings,
        error_to_raise=RuntimeError("Ollama daemon unreachable or crashed"),
    )
    openai_mock = MockProvider(name="openai", response_content="OpenAI should not run")

    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_provider,
        openai_provider=openai_mock,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Should fail cleanly")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    with pytest.raises(RuntimeError, match="Ollama daemon unreachable"):
        await orchestrator.handle_chat_request(request)

    assert len(recorded) == 1
    # OpenAI must NEVER be invoked as an automatic runtime fallback
    assert openai_mock.call_count == 0


@pytest.mark.anyio
async def test_openai_runtime_failure_does_not_invoke_ollama(
    both_configured_settings: Settings,
) -> None:
    openai_mock = MockProvider(
        name="openai",
        error_to_raise=RuntimeError("OpenAI API connection failed"),
    )
    ollama_provider, recorded = make_mock_ollama_provider(
        both_configured_settings, response_content="Ollama should not run"
    )

    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_provider,
        openai_provider=openai_mock,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Cloud request")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        preferred_provider=ProviderName.OPENAI,
    )

    with pytest.raises(RuntimeError, match="OpenAI API connection failed"):
        await orchestrator.handle_chat_request(request)

    assert openai_mock.call_count == 1
    # Ollama must NEVER be invoked on OpenAI failure
    assert len(recorded) == 0


# ---------------------------------------------------------------------------
# Requirement 10: Provider identity mismatch is handled safely
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_provider_identity_mismatch_handled_safely(
    both_configured_settings: Settings,
) -> None:
    # A real OllamaProvider subclass returning mismatched response metadata
    class MismatchedOllamaProvider(OllamaProvider):
        async def generate(self, messages: Sequence[ChatMessage]) -> ProviderResponse:
            return ProviderResponse(
                content="Model response",
                provider="untrusted-vendor",
                model="alien-model-v99",
            )

    ollama_mismatch = MismatchedOllamaProvider(settings=both_configured_settings)

    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_mismatch,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Check identity")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    response = await orchestrator.handle_chat_request(request)

    # Response must strictly reflect the authoritative routing decision
    assert response.provider == ProviderName.OLLAMA
    assert response.model == "llama3.2:latest"
    # Identity mismatch safely documented in metadata
    assert response.metadata.get("identity_mismatch") is True
    assert response.metadata.get("provider_reported_name") == "untrusted-vendor"
    assert response.metadata.get("model_mismatch") is True
    assert response.metadata.get("provider_reported_model") == "alien-model-v99"


# ---------------------------------------------------------------------------
# Requirement 11: Empty or invalid provider output is handled appropriately
# ---------------------------------------------------------------------------
@pytest.mark.anyio
@pytest.mark.parametrize(
    "bad_content",
    [
        "",
        "   ",
        "\n\t  \r",
    ],
)
async def test_empty_or_whitespace_provider_output(
    both_configured_settings: Settings,
    bad_content: str,
) -> None:
    ollama_provider, _ = make_mock_ollama_provider(
        both_configured_settings,
        response_content=bad_content,
    )
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_provider,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Expect answer")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    with pytest.raises(ValueError, match="empty or whitespace-only content"):
        await orchestrator.handle_chat_request(request)


@pytest.mark.anyio
async def test_invalid_provider_response_type(
    both_configured_settings: Settings,
) -> None:
    class BadReturnOllama(OllamaProvider):
        async def generate(self, messages: Sequence[ChatMessage]) -> Any:
            return {"not": "a ProviderResponse"}

    ollama_bad = BadReturnOllama(settings=both_configured_settings)
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_bad,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Expect answer")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    with pytest.raises(ValueError, match="invalid response object"):
        await orchestrator.handle_chat_request(request)


# ---------------------------------------------------------------------------
# Requirement 12: Provider injection works without live network calls
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_provider_injection_without_live_network(
    none_configured_settings: Settings,
) -> None:
    # Injected OllamaProvider with safe MockTransport executes in-memory
    custom_settings = Settings(
        OPENAI_API_KEY=None,
        OPENAI_MODEL=None,
        OLLAMA_MODEL="custom-mock-model",
        OLLAMA_BASE_URL="http://127.0.0.1:11434",
    )
    ollama_provider, recorded = make_mock_ollama_provider(
        custom_settings,
        response_content="In-memory test pass",
        response_model="custom-mock-model",
    )

    orchestrator = ChatOrchestrator(
        settings=custom_settings,
        providers={ProviderName.OLLAMA: ollama_provider},
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="No network request")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    response = await orchestrator.handle_chat_request(request)
    assert response.content == "In-memory test pass"
    assert response.provider == ProviderName.OLLAMA
    assert response.model == "custom-mock-model"
    assert len(recorded) == 1


# ---------------------------------------------------------------------------
# Orchestrator alias and lazy initialization tests
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_orchestrator_alias_method(both_configured_settings: Settings) -> None:
    ollama_provider, recorded = make_mock_ollama_provider(
        both_configured_settings, response_content="Alias test"
    )
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_provider,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Call alias")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    response = await orchestrator.orchestrate(request)
    assert response.content == "Alias test"
    assert len(recorded) == 1


def test_lazy_provider_resolution() -> None:
    settings = Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="gpt-4o-mini",
        OLLAMA_MODEL="llama3.2:latest",
    )
    orchestrator = ChatOrchestrator(settings=settings)

    # Lazy instantiation of OllamaProvider
    ollama = orchestrator._get_provider(ProviderName.OLLAMA)
    assert ollama.name == "ollama"

    # Lazy instantiation of OpenAIProvider
    openai = orchestrator._get_provider(ProviderName.OPENAI)
    assert openai.name == "openai"

    # Re-accessing returns the cached instance
    assert orchestrator._get_provider(ProviderName.OLLAMA) is ollama
    assert orchestrator._get_provider(ProviderName.OPENAI) is openai


# ---------------------------------------------------------------------------
# Finding 1 Tests: LOCAL_ONLY Destination Boundary Enforcement
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_local_only_default_local_endpoint_accepted(
    both_configured_settings: Settings,
) -> None:
    ollama_provider, recorded = make_mock_ollama_provider(
        both_configured_settings, response_content="OK"
    )
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama_provider,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Test")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    resp = await orchestrator.handle_chat_request(request)
    assert resp.content == "OK"
    assert len(recorded) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "valid_local_url",
    [
        "http://localhost:11434",
        "http://localhost:11434/",
        "http://localhost",
        "http://localhost.",
        "https://localhost:11434",
        "http://127.0.0.1:11434",
        "http://127.0.0.1",
        "http://127.0.0.2:11434",
        "http://[::1]:11434",
        "http://[::1]",
    ],
)
async def test_local_only_accepted_loopback_destinations(
    valid_local_url: str,
) -> None:
    settings = Settings(
        OLLAMA_MODEL="llama3.2:latest",
        OLLAMA_BASE_URL=valid_local_url,
    )
    ollama_provider, recorded = make_mock_ollama_provider(
        settings, response_content="Loopback OK"
    )
    orchestrator = ChatOrchestrator(
        settings=settings,
        ollama_provider=ollama_provider,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Test loopback")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    resp = await orchestrator.handle_chat_request(request)
    assert resp.content == "Loopback OK"
    assert len(recorded) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "remote_or_invalid_url",
    [
        "http://remote.host:11434",
        "http://api.ollama.ai:11434",
        "http://192.168.1.50:11434",
        "http://10.0.0.1:11434",
        "http://8.8.8.8:11434",
        "http://localhost.evil.com:11434",
        "http://fake-localhost:11434",
        "http://notlocalhost:11434",
        "http://user:pass@127.0.0.1:11434",
        "ftp://127.0.0.1:11434",
        "not_a_valid_url",
        "://invalid",
        "",
    ],
)
async def test_local_only_rejects_remote_and_invalid_destinations(
    remote_or_invalid_url: str,
) -> None:
    settings = Settings(
        OLLAMA_MODEL="llama3.2:latest",
        OLLAMA_BASE_URL=remote_or_invalid_url if remote_or_invalid_url else "http://127.0.0.1:11434",
    )
    ollama_provider, recorded = make_mock_ollama_provider(
        settings, response_content="Should not be called"
    )
    if remote_or_invalid_url == "":
        ollama_provider._settings = Settings(
            OLLAMA_MODEL="llama3.2:latest", OLLAMA_BASE_URL=""
        )

    orchestrator = ChatOrchestrator(
        settings=settings,
        ollama_provider=ollama_provider,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Private content")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    with pytest.raises(RoutingPolicyError, match="LOCAL_ONLY destination boundary violation"):
        await orchestrator.handle_chat_request(request)

    # Provider must NEVER be invoked when destination validation fails
    assert len(recorded) == 0


@pytest.mark.anyio
async def test_cloud_allowed_permits_remote_ollama_destination() -> None:
    remote_url = "http://remote-team-ollama.internal:11434"
    settings = Settings(
        OLLAMA_MODEL="llama3.2:latest",
        OLLAMA_BASE_URL=remote_url,
    )
    ollama_provider, recorded = make_mock_ollama_provider(
        settings, response_content="Remote Ollama OK"
    )

    orchestrator = ChatOrchestrator(
        settings=settings,
        ollama_provider=ollama_provider,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Cloud allowed request")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        preferred_provider=ProviderName.OLLAMA,
    )

    resp = await orchestrator.handle_chat_request(request)
    assert resp.content == "Remote Ollama OK"
    assert len(recorded) == 1


# ---------------------------------------------------------------------------
# Finding 2 Tests: Protect Orchestrator-Owned Metadata
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "malicious_key",
    [
        "identity_mismatch",
        "model_mismatch",
        "provider_reported_name",
        "provider_reported_model",
        "IDENTITY_MISMATCH",
        "Provider_Reported_Name",
    ],
)
def test_client_metadata_rejects_orchestrator_diagnostic_keys(
    malicious_key: str,
) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Client metadata contains reserved system key"):
        ChatRequest(
            messages=[ChatMessageInput(role="user", content="Hello")],
            metadata={malicious_key: True},
        )


@pytest.mark.anyio
async def test_normal_metadata_pass_through_and_diagnostic_integrity(
    both_configured_settings: Settings,
) -> None:
    class MismatchedMetadataOllama(OllamaProvider):
        async def generate(self, messages: Sequence[ChatMessage]) -> ProviderResponse:
            return ProviderResponse(
                content="Response",
                provider="mismatched_provider_name",
                model="mismatched_model_name",
            )

    ollama = MismatchedMetadataOllama(settings=both_configured_settings)
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Hello")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
        metadata={
            "client_tracking_id": "track-456",
            "correlation_id": "corr-789",
        },
    )

    response = await orchestrator.handle_chat_request(request)

    # Legitimate client metadata is preserved
    assert response.metadata["client_tracking_id"] == "track-456"
    assert response.metadata["correlation_id"] == "corr-789"

    # Orchestrator-owned diagnostics are accurately populated
    assert response.metadata["identity_mismatch"] is True
    assert response.metadata["model_mismatch"] is True
    assert response.metadata["provider_reported_name"] == "mismatched_provider_name"
    assert response.metadata["provider_reported_model"] == "mismatched_model_name"

    # Authoritative provider and model reflect DEV routing, not provider self-reporting
    assert response.provider == ProviderName.OLLAMA
    assert response.model == "llama3.2:latest"


@pytest.mark.anyio
async def test_orchestrator_local_only_proxy_environment_isolation(
    both_configured_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Set hostile proxy env vars attempting to redirect traffic
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.hostile.corp:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.hostile.corp:8080")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.hostile.corp:8080")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Local secure reply"}},
        )

    # OllamaProvider with trust_env=False and mock transport
    from backend.app.providers.ollama_provider import OllamaProvider

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        trust_env=False,
    ) as client:
        real_ollama = OllamaProvider(
            settings=both_configured_settings,
            http_client=client,
            trust_env=False,
        )
        orchestrator = ChatOrchestrator(
            settings=both_configured_settings,
            ollama_provider=real_ollama,
        )

        request = ChatRequest(
            messages=[ChatMessageInput(role="user", content="Secret private prompt")],
            privacy_level=PrivacyLevel.LOCAL_ONLY,
        )

        response = await orchestrator.handle_chat_request(request)
        assert response.content == "Local secure reply"
        assert response.provider == ProviderName.OLLAMA


# ---------------------------------------------------------------------------
# Provider Identity Enforcement & Proxy Transport Hardening Tests
# ---------------------------------------------------------------------------
def test_validate_provider_identity_helper(both_configured_settings: Settings) -> None:
    from backend.app.orchestrator.service import validate_provider_identity
    from backend.app.providers.ollama_provider import OllamaProvider
    from backend.app.providers.openai_provider import OpenAIProvider

    real_openai = OpenAIProvider(settings=both_configured_settings)
    real_ollama = OllamaProvider(settings=both_configured_settings)

    # Valid identities pass
    validate_provider_identity(ProviderName.OLLAMA, real_ollama)
    validate_provider_identity(ProviderName.OPENAI, real_openai)
    validate_provider_identity(ProviderName.OLLAMA, MockProvider(name="ollama"))
    validate_provider_identity(ProviderName.OPENAI, MockProvider(name="openai"))

    # Mismatches raise RoutingPolicyError
    with pytest.raises(RoutingPolicyError, match="Provider identity mismatch: OpenAIProvider cannot be"):
        validate_provider_identity(ProviderName.OLLAMA, real_openai)

    with pytest.raises(RoutingPolicyError, match="Provider identity mismatch: OllamaProvider cannot be"):
        validate_provider_identity(ProviderName.OPENAI, real_ollama)

    with pytest.raises(RoutingPolicyError, match="Provider identity mismatch: provider with name 'anthropic'"):
        validate_provider_identity(ProviderName.OLLAMA, MockProvider(name="anthropic"))

    with pytest.raises(RoutingPolicyError, match="Provider identity mismatch: provider with name 'anthropic'"):
        validate_provider_identity(ProviderName.OPENAI, MockProvider(name="anthropic"))


def test_orchestrator_rejects_mismatched_provider_injection_at_init(
    both_configured_settings: Settings,
) -> None:
    from backend.app.providers.ollama_provider import OllamaProvider
    from backend.app.providers.openai_provider import OpenAIProvider

    real_openai = OpenAIProvider(settings=both_configured_settings)
    real_ollama = OllamaProvider(settings=both_configured_settings)

    # Injecting OpenAI as ollama_provider rejected
    with pytest.raises(RoutingPolicyError, match="Provider identity mismatch: OpenAIProvider cannot be"):
        ChatOrchestrator(
            settings=both_configured_settings,
            ollama_provider=real_openai,
        )

    # Injecting Ollama as openai_provider rejected
    with pytest.raises(RoutingPolicyError, match="Provider identity mismatch: OllamaProvider cannot be"):
        ChatOrchestrator(
            settings=both_configured_settings,
            openai_provider=real_ollama,
        )

    # Injecting mismatched mapping rejected
    with pytest.raises(RoutingPolicyError, match="Provider identity mismatch: OpenAIProvider cannot be"):
        ChatOrchestrator(
            settings=both_configured_settings,
            providers={ProviderName.OLLAMA: real_openai},
        )


@pytest.mark.anyio
async def test_orchestrator_rejects_injected_provider_with_unsafe_proxy(
    both_configured_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.providers.ollama_provider import OllamaProvider

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.leak.corp:8080")
    unsafe_client = httpx.AsyncClient(trust_env=True)

    unsafe_ollama = OllamaProvider(
        settings=both_configured_settings,
        http_client=unsafe_client,
    )
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=unsafe_ollama,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Classified info")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    with pytest.raises(ProviderConfigurationError, match="proxy"):
        await orchestrator.handle_chat_request(request)


@pytest.mark.anyio
async def test_remote_ollama_with_cloud_allowed_retains_proxy_compatibility() -> None:
    from backend.app.providers.ollama_provider import OllamaProvider

    settings = Settings(
        OLLAMA_BASE_URL="http://remote-team-ollama.internal:11434",
        OLLAMA_MODEL="llama3.2:latest",
    )
    orchestrator = ChatOrchestrator(settings=settings)

    # Lazy instantiation under remote endpoint has trust_env=True dynamically
    prov = orchestrator._get_provider(ProviderName.OLLAMA)
    assert isinstance(prov, OllamaProvider)
    assert prov._trust_env is None


# ---------------------------------------------------------------------------
# Privacy Boundary & LOCAL_ONLY Security Hardening Tests
# ---------------------------------------------------------------------------
def test_validate_provider_identity_rejects_missing_or_empty_name() -> None:
    from backend.app.orchestrator.service import validate_provider_identity

    class NamelessProvider:
        pass

    class EmptyNameProvider:
        name = "   "

    with pytest.raises(RoutingPolicyError, match="missing a valid 'name' attribute"):
        validate_provider_identity(ProviderName.OLLAMA, NamelessProvider())

    with pytest.raises(RoutingPolicyError, match="missing a valid 'name' attribute"):
        validate_provider_identity(ProviderName.OPENAI, EmptyNameProvider())


@pytest.mark.anyio
async def test_proxy_configured_injected_transport_cannot_receive_local_only_prompts(
    both_configured_settings: Settings,
) -> None:
    proxy_transport = httpx.AsyncHTTPTransport(
        proxy="http://proxy.leak.corp:8080"
    )
    proxy_client = httpx.AsyncClient(transport=proxy_transport, trust_env=False)
    ollama = OllamaProvider(
        settings=both_configured_settings,
        http_client=proxy_client,
    )
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Top secret local info")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    with pytest.raises(ProviderConfigurationError, match="explicit proxy transport configured"):
        await orchestrator.handle_chat_request(request)


@pytest.mark.anyio
async def test_unverifiable_injected_transport_rejected_before_execution(
    both_configured_settings: Settings,
) -> None:
    class CustomUnverifiableTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "exfiltrated"}},
            )

    unverifiable_client = httpx.AsyncClient(
        transport=CustomUnverifiableTransport(),
        trust_env=False,
    )
    ollama = OllamaProvider(
        settings=both_configured_settings,
        http_client=unverifiable_client,
    )
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=ollama,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Top secret local info")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    with pytest.raises(ProviderConfigurationError, match="custom or unverifiable transport"):
        await orchestrator.handle_chat_request(request)


@pytest.mark.anyio
async def test_custom_provider_impersonating_ollama_cannot_execute_local_only(
    both_configured_settings: Settings,
) -> None:
    class FakeOllamaProvider:
        name = "ollama"
        base_url = "http://localhost:11434"

        async def generate(self, messages: Sequence[ChatMessage]) -> ProviderResponse:
            return ProviderResponse(
                content="exfiltrated data",
                provider="ollama",
                model="llama3.2:latest",
            )

    # Impersonator passes __init__ key validation because name == "ollama",
    # but MUST be rejected at execution for LOCAL_ONLY requests
    fake_ollama = FakeOllamaProvider()
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=fake_ollama,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Top secret local info")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    with pytest.raises(
        RoutingPolicyError,
        match="LOCAL_ONLY requests require a verified OllamaProvider adapter",
    ):
        await orchestrator.handle_chat_request(request)


@pytest.mark.anyio
async def test_injected_provider_own_endpoint_validated_not_orchestrator_settings() -> None:
    # Orchestrator settings specify a safe local endpoint,
    # but the injected OllamaProvider is configured to target a remote server.
    orchestrator_settings = Settings(
        OLLAMA_BASE_URL="http://localhost:11434",
        OLLAMA_MODEL="llama3.2:latest",
    )
    remote_provider_settings = Settings(
        OLLAMA_BASE_URL="http://remote.hostile.internal:11434",
        OLLAMA_MODEL="llama3.2:latest",
    )
    remote_ollama = OllamaProvider(settings=remote_provider_settings)
    orchestrator = ChatOrchestrator(
        settings=orchestrator_settings,
        ollama_provider=remote_ollama,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Top secret local info")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    # Must fail because the injected provider's own base_url is remote,
    # proving the orchestrator does not infer the endpoint from its own settings.
    with pytest.raises(
        RoutingPolicyError,
        match="LOCAL_ONLY destination boundary violation: host 'remote.hostile.internal'",
    ):
        await orchestrator.handle_chat_request(request)


@pytest.mark.anyio
async def test_injected_provider_without_base_url_rejected_for_local_only(
    both_configured_settings: Settings,
) -> None:
    class NoBaseUrlOllama(OllamaProvider):
        @property
        def base_url(self) -> Any:
            return None

    no_url_ollama = NoBaseUrlOllama(settings=both_configured_settings)
    orchestrator = ChatOrchestrator(
        settings=both_configured_settings,
        ollama_provider=no_url_ollama,
    )

    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Top secret local info")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )

    with pytest.raises(
        RoutingPolicyError,
        match="LOCAL_ONLY destination boundary violation: local provider endpoint is not configured",
    ):
        await orchestrator.handle_chat_request(request)

