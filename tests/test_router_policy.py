"""Unit tests for the DEV deterministic AI routing policy."""

import pytest

from backend.app.config import Settings
from backend.app.router.enums import PrivacyLevel, ProviderName, TaskComplexity
from backend.app.router.policy import (
    RoutingPolicyError,
    is_ollama_configured,
    is_openai_configured,
    route_request,
)
from backend.app.router.schemas import ChatMessageInput, ChatRequest


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


def test_configuration_readiness_helper(
    both_configured_settings: Settings,
    ollama_only_settings: Settings,
    openai_only_settings: Settings,
    none_configured_settings: Settings,
) -> None:
    assert is_ollama_configured(both_configured_settings) is True
    assert is_openai_configured(both_configured_settings) is True

    assert is_ollama_configured(ollama_only_settings) is True
    assert is_openai_configured(ollama_only_settings) is False

    assert is_ollama_configured(openai_only_settings) is False
    assert is_openai_configured(openai_only_settings) is True

    assert is_ollama_configured(none_configured_settings) is False
    assert is_openai_configured(none_configured_settings) is False


def test_local_only_requests_select_ollama(both_configured_settings: Settings) -> None:
    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Local prompt")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
    )
    decision = route_request(request, both_configured_settings)

    assert decision.selected_provider == ProviderName.OLLAMA
    assert decision.model == "llama3.2:latest"
    assert decision.is_fallback is False
    assert decision.fallback_reason is None
    assert "local_only" in decision.reason


def test_local_only_requests_never_select_openai(both_configured_settings: Settings) -> None:
    for complexity in [TaskComplexity.SIMPLE, TaskComplexity.COMPLEX, TaskComplexity.AUTO]:
        request = ChatRequest(
            messages=[ChatMessageInput(role="user", content="Prompt")],
            privacy_level=PrivacyLevel.LOCAL_ONLY,
            complexity=complexity,
            allow_fallback=True,
        )
        decision = route_request(request, both_configured_settings)
        assert decision.selected_provider != ProviderName.OPENAI
        assert decision.selected_provider == ProviderName.OLLAMA


def test_explicit_openai_with_local_only_rejected(both_configured_settings: Settings) -> None:
    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Prompt")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
        preferred_provider=ProviderName.OPENAI,
    )
    with pytest.raises(
        RoutingPolicyError,
        match="Explicit provider preference 'openai' violates privacy level 'local_only'",
    ):
        route_request(request, both_configured_settings)


def test_simple_tasks_prefer_ollama(both_configured_settings: Settings) -> None:
    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Simple math 2+2")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        complexity=TaskComplexity.SIMPLE,
    )
    decision = route_request(request, both_configured_settings)

    assert decision.selected_provider == ProviderName.OLLAMA
    assert decision.model == "llama3.2:latest"
    assert decision.is_fallback is False


def test_complex_cloud_allowed_tasks_prefer_openai(both_configured_settings: Settings) -> None:
    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Deep architecture analysis")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        complexity=TaskComplexity.COMPLEX,
    )
    decision = route_request(request, both_configured_settings)

    assert decision.selected_provider == ProviderName.OPENAI
    assert decision.model == "gpt-4o-mini"
    assert decision.is_fallback is False
    assert "complex" in decision.reason


def test_complex_local_only_tasks_remain_on_ollama(both_configured_settings: Settings) -> None:
    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Complex private analysis")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
        complexity=TaskComplexity.COMPLEX,
    )
    decision = route_request(request, both_configured_settings)

    assert decision.selected_provider == ProviderName.OLLAMA
    assert decision.model == "llama3.2:latest"
    assert decision.is_fallback is False


def test_auto_complexity_prefers_ollama(both_configured_settings: Settings) -> None:
    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Hello")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        complexity=TaskComplexity.AUTO,
    )
    decision = route_request(request, both_configured_settings)

    assert decision.selected_provider == ProviderName.OLLAMA
    assert decision.model == "llama3.2:latest"
    assert decision.is_fallback is False


def test_explicit_provider_preferences_respected(both_configured_settings: Settings) -> None:
    req_ollama = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Run local")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        preferred_provider=ProviderName.OLLAMA,
    )
    dec_ollama = route_request(req_ollama, both_configured_settings)
    assert dec_ollama.selected_provider == ProviderName.OLLAMA
    assert dec_ollama.is_fallback is False

    req_openai = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Run cloud")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        preferred_provider=ProviderName.OPENAI,
    )
    dec_openai = route_request(req_openai, both_configured_settings)
    assert dec_openai.selected_provider == ProviderName.OPENAI
    assert dec_openai.is_fallback is False


def test_missing_provider_configuration_produces_clear_error(
    none_configured_settings: Settings,
    ollama_only_settings: Settings,
    openai_only_settings: Settings,
) -> None:
    # Neither configured
    request_auto = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Hello")],
        allow_fallback=False,
    )
    with pytest.raises(RoutingPolicyError, match="not configured"):
        route_request(request_auto, none_configured_settings)

    # Explicit OpenAI requested but not configured
    request_openai = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Cloud please")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        preferred_provider=ProviderName.OPENAI,
        allow_fallback=False,
    )
    with pytest.raises(RoutingPolicyError, match="Provider 'openai' was explicitly requested but is not configured"):
        route_request(request_openai, ollama_only_settings)

    # Explicit Ollama requested but not configured
    request_ollama = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Local please")],
        preferred_provider=ProviderName.OLLAMA,
        allow_fallback=False,
    )
    with pytest.raises(RoutingPolicyError, match="Provider 'ollama' was explicitly requested but is not configured"):
        route_request(request_ollama, openai_only_settings)


def test_fallback_disabled_requests_never_switch_providers(
    ollama_only_settings: Settings,
    openai_only_settings: Settings,
) -> None:
    # Complex cloud-allowed task prefers OpenAI, but OpenAI is missing; fallback disabled
    request_complex = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Complex task")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        complexity=TaskComplexity.COMPLEX,
        allow_fallback=False,
    )
    with pytest.raises(RoutingPolicyError, match="fallback is disabled"):
        route_request(request_complex, ollama_only_settings)

    # Simple task prefers Ollama, but Ollama is missing; fallback disabled
    request_simple = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Simple task")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        complexity=TaskComplexity.SIMPLE,
        allow_fallback=False,
    )
    with pytest.raises(RoutingPolicyError, match="fallback is disabled"):
        route_request(request_simple, openai_only_settings)


def test_allowed_fallback_selects_eligible_alternative(
    ollama_only_settings: Settings,
    openai_only_settings: Settings,
) -> None:
    # Preferred OpenAI is missing; falls back to Ollama
    request_cloud = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Complex task")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        complexity=TaskComplexity.COMPLEX,
        allow_fallback=True,
    )
    decision = route_request(request_cloud, ollama_only_settings)
    assert decision.selected_provider == ProviderName.OLLAMA
    assert decision.model == "llama3.2:latest"
    assert decision.is_fallback is True
    assert decision.fallback_reason is not None
    assert "openai' is not configured" in decision.fallback_reason
    assert decision.evaluated_providers == [ProviderName.OPENAI, ProviderName.OLLAMA]

    # Preferred Ollama is missing; cloud_allowed task falls back to OpenAI
    request_local_eligible = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Simple task")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        complexity=TaskComplexity.SIMPLE,
        allow_fallback=True,
    )
    decision_openai = route_request(request_local_eligible, openai_only_settings)
    assert decision_openai.selected_provider == ProviderName.OPENAI
    assert decision_openai.model == "gpt-4o-mini"
    assert decision_openai.is_fallback is True
    assert decision_openai.fallback_reason is not None
    assert "ollama' is not configured" in decision_openai.fallback_reason
    assert decision_openai.evaluated_providers == [ProviderName.OLLAMA, ProviderName.OPENAI]


def test_local_only_requests_never_fall_back_to_openai(openai_only_settings: Settings) -> None:
    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Strictly local task")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
        allow_fallback=True,
    )
    with pytest.raises(
        RoutingPolicyError,
        match="fallback to cloud provider 'openai' is strictly prohibited under 'local_only' privacy",
    ):
        route_request(request, openai_only_settings)


def test_routing_decision_contains_correct_model_and_explanation(
    both_configured_settings: Settings,
) -> None:
    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Explain relativity")],
        privacy_level=PrivacyLevel.CLOUD_ALLOWED,
        complexity=TaskComplexity.COMPLEX,
    )
    decision = route_request(request, both_configured_settings)

    assert decision.selected_provider == ProviderName.OPENAI
    assert decision.model == "gpt-4o-mini"
    assert isinstance(decision.reason, str)
    assert len(decision.reason) > 0
    assert decision.is_fallback is False
    assert decision.fallback_reason is None
    assert decision.evaluated_providers == [ProviderName.OPENAI]


def test_auto_is_never_returned_as_selected_provider(both_configured_settings: Settings) -> None:
    for priv in [PrivacyLevel.LOCAL_ONLY, PrivacyLevel.CLOUD_ALLOWED]:
        for comp in [TaskComplexity.AUTO, TaskComplexity.SIMPLE, TaskComplexity.COMPLEX]:
            request = ChatRequest(
                messages=[ChatMessageInput(role="user", content="Testing auto absence")],
                privacy_level=priv,
                complexity=comp,
                preferred_provider=ProviderName.AUTO,
            )
            decision = route_request(request, both_configured_settings)
            assert decision.selected_provider in (ProviderName.OLLAMA, ProviderName.OPENAI)
            assert decision.selected_provider != ProviderName.AUTO


def test_untrusted_client_metadata_does_not_affect_routing(both_configured_settings: Settings) -> None:
    # Metadata includes arbitrary client tracking data
    request = ChatRequest(
        messages=[ChatMessageInput(role="user", content="Prompt")],
        privacy_level=PrivacyLevel.LOCAL_ONLY,
        metadata={"client_agent": "mobile_app_v1", "correlation_id": "corr-42"},
    )
    decision = route_request(request, both_configured_settings)

    assert decision.selected_provider == ProviderName.OLLAMA
    # Ensure metadata was not forwarded or leaked into the authority decision metadata
    assert "client_agent" not in decision.metadata
