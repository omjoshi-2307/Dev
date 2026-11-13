"""Deterministic AI routing policy for DEV."""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.app.router.enums import PrivacyLevel, ProviderName, TaskComplexity
from backend.app.router.schemas import ChatRequest, ConcreteProvider, RoutingDecision

if TYPE_CHECKING:
    from backend.app.config import Settings


class RoutingPolicyError(RuntimeError):
    """Raised when a request cannot be routed safely under the active policy."""


def is_ollama_configured(settings: Settings) -> bool:
    """Return True if Ollama has a configured model name.

    Note: Configuration readiness indicates that required settings are defined;
    it does not guarantee live runtime availability of the Ollama daemon.
    """
    return bool(settings.ollama_model and settings.ollama_model.strip())


def is_openai_configured(settings: Settings) -> bool:
    """Return True if OpenAI has both an API key and a model configured.

    Note: Configuration readiness indicates that required settings are defined;
    it does not guarantee live network reachability or credential validity.
    """
    return bool(
        settings.openai_api_key
        and settings.openai_api_key.strip()
        and settings.openai_model
        and settings.openai_model.strip()
    )


def route_request(
    request: ChatRequest,
    settings: Settings | None = None,
) -> RoutingDecision:
    """Deterministically select a provider and model for a validated ChatRequest.

    Rules enforced:
    1. Privacy: Under LOCAL_ONLY, OpenAI is strictly prohibited, including via fallback.
       Explicit client requests for OpenAI under LOCAL_ONLY immediately raise RoutingPolicyError.
       An inherited OpenAI default from environment configuration is NOT treated as an explicit
       client request; under LOCAL_ONLY, it routes to Ollama per automatic local policy.
    2. Explicit Preference: When explicitly specified by the client (OLLAMA or OPENAI), it is respected.
       Fallback to an alternative is permitted only when allow_fallback is True and
       the alternative satisfies privacy constraints.
    3. Automatic Selection (AUTO or inherited default):
       - Under LOCAL_ONLY, always prefers local Ollama.
       - Under CLOUD_ALLOWED:
         - COMPLEX -> prefers OpenAI.
         - SIMPLE or AUTO -> prefers Ollama (cost/latency optimization).
         - Inherited OpenAI configuration default -> prefers OpenAI.
         - Inherited Ollama configuration default -> prefers Ollama.
    4. Readiness: A provider is eligible only if required configuration settings exist.
       Without allow_fallback, an unconfigured preferred provider raises RoutingPolicyError.
    """
    if settings is None:
        from backend.app.config import get_settings

        active_settings = get_settings()
    else:
        active_settings = settings

    # Distinguish an explicit client preference from a default-factory inherited setting
    is_explicit_preference = (
        "preferred_provider" in request.model_fields_set
        and request.preferred_provider != ProviderName.AUTO
    )

    # Branch A: Explicit Client Provider Preference
    if is_explicit_preference:
        if request.preferred_provider == ProviderName.OPENAI:
            if request.privacy_level == PrivacyLevel.LOCAL_ONLY:
                raise RoutingPolicyError(
                    "Explicit provider preference 'openai' violates privacy level 'local_only'. "
                    "Cloud processing is strictly prohibited."
                )

            evaluated: list[ConcreteProvider] = [ProviderName.OPENAI]
            if is_openai_configured(active_settings):
                assert active_settings.openai_model is not None
                return RoutingDecision(
                    selected_provider=ProviderName.OPENAI,
                    model=active_settings.openai_model,
                    reason="Explicit user preference for OpenAI under cloud_allowed privacy.",
                    is_fallback=False,
                    fallback_reason=None,
                    evaluated_providers=evaluated,
                )

            # OpenAI is not configured
            if not request.allow_fallback:
                raise RoutingPolicyError(
                    "Provider 'openai' was explicitly requested but is not configured "
                    "(missing OPENAI_API_KEY or OPENAI_MODEL)."
                )

            evaluated.append(ProviderName.OLLAMA)
            if is_ollama_configured(active_settings):
                assert active_settings.ollama_model is not None
                return RoutingDecision(
                    selected_provider=ProviderName.OLLAMA,
                    model=active_settings.ollama_model,
                    reason="Explicit provider 'openai' is not configured; fell back to 'ollama'.",
                    is_fallback=True,
                    fallback_reason=(
                        "Explicitly preferred provider 'openai' is not configured "
                        "(missing OPENAI_API_KEY or OPENAI_MODEL)."
                    ),
                    evaluated_providers=evaluated,
                )

            raise RoutingPolicyError(
                "Explicitly requested provider 'openai' is not configured, and fallback "
                "provider 'ollama' is also not configured."
            )

        if request.preferred_provider == ProviderName.OLLAMA:
            evaluated = [ProviderName.OLLAMA]
            if is_ollama_configured(active_settings):
                assert active_settings.ollama_model is not None
                return RoutingDecision(
                    selected_provider=ProviderName.OLLAMA,
                    model=active_settings.ollama_model,
                    reason=f"Explicit user preference for Ollama ({request.privacy_level.value}).",
                    is_fallback=False,
                    fallback_reason=None,
                    evaluated_providers=evaluated,
                )

            # Ollama is not configured
            if not request.allow_fallback:
                raise RoutingPolicyError(
                    "Provider 'ollama' was explicitly requested but is not configured "
                    "(missing OLLAMA_MODEL)."
                )

            # Check if fallback to OpenAI is permissible
            if request.privacy_level == PrivacyLevel.LOCAL_ONLY:
                raise RoutingPolicyError(
                    "Provider 'ollama' is not configured, and fallback to cloud provider 'openai' "
                    "is strictly prohibited under 'local_only' privacy."
                )

            evaluated.append(ProviderName.OPENAI)
            if is_openai_configured(active_settings):
                assert active_settings.openai_model is not None
                return RoutingDecision(
                    selected_provider=ProviderName.OPENAI,
                    model=active_settings.openai_model,
                    reason=(
                        "Explicit provider 'ollama' is not configured; fell back to 'openai' "
                        "under cloud_allowed privacy."
                    ),
                    is_fallback=True,
                    fallback_reason="Explicitly preferred provider 'ollama' is not configured (missing OLLAMA_MODEL).",
                    evaluated_providers=evaluated,
                )

            raise RoutingPolicyError(
                "Explicitly requested provider 'ollama' is not configured, and fallback "
                "provider 'openai' is also not configured."
            )

    # Branch B: Automatic Provider Selection & Inherited Defaults
    # When preferred_provider is not an explicit client preference:
    # 1. If privacy is LOCAL_ONLY, cloud is strictly prohibited; candidate is always Ollama.
    # 2. If privacy is CLOUD_ALLOWED, respect configured default provider if specific,
    #    or evaluate complexity if AUTO.
    if request.privacy_level == PrivacyLevel.LOCAL_ONLY:
        primary_candidate: ConcreteProvider = ProviderName.OLLAMA
        primary_reason = (
            f"Privacy level is 'local_only'; routes to local Ollama "
            f"(complexity: {request.complexity.value})."
        )
    elif request.preferred_provider == ProviderName.OPENAI:
        # Inherited configuration default is OpenAI under cloud_allowed privacy
        primary_candidate = ProviderName.OPENAI
        primary_reason = "Configured default provider is OpenAI under cloud_allowed privacy."
    elif request.preferred_provider == ProviderName.OLLAMA:
        # Inherited configuration default is Ollama
        primary_candidate = ProviderName.OLLAMA
        primary_reason = "Configured default provider is Ollama."
    elif request.complexity == TaskComplexity.COMPLEX:
        primary_candidate = ProviderName.OPENAI
        primary_reason = (
            "Task complexity is 'complex' with 'cloud_allowed' privacy; prefers OpenAI."
        )
    else:
        # SIMPLE or AUTO complexity under CLOUD_ALLOWED
        primary_candidate = ProviderName.OLLAMA
        primary_reason = (
            f"Task complexity is '{request.complexity.value}'; prefers local Ollama "
            "for cost efficiency and data locality."
        )

    # Evaluate primary candidate readiness
    if primary_candidate == ProviderName.OPENAI:
        evaluated = [ProviderName.OPENAI]
        if is_openai_configured(active_settings):
            assert active_settings.openai_model is not None
            return RoutingDecision(
                selected_provider=ProviderName.OPENAI,
                model=active_settings.openai_model,
                reason=primary_reason,
                is_fallback=False,
                fallback_reason=None,
                evaluated_providers=evaluated,
            )

        if not request.allow_fallback:
            raise RoutingPolicyError(
                "Preferred cloud provider 'openai' is not configured "
                "(missing OPENAI_API_KEY or OPENAI_MODEL) and fallback is disabled."
            )

        evaluated.append(ProviderName.OLLAMA)
        if is_ollama_configured(active_settings):
            assert active_settings.ollama_model is not None
            return RoutingDecision(
                selected_provider=ProviderName.OLLAMA,
                model=active_settings.ollama_model,
                reason="Preferred provider 'openai' is not configured; fell back to local 'ollama'.",
                is_fallback=True,
                fallback_reason=(
                    "Preferred cloud provider 'openai' is not configured "
                    "(missing OPENAI_API_KEY or OPENAI_MODEL)."
                ),
                evaluated_providers=evaluated,
            )

        raise RoutingPolicyError(
            "Preferred cloud provider 'openai' is not configured, and fallback "
            "provider 'ollama' is also not configured."
        )

    # primary_candidate == ProviderName.OLLAMA
    evaluated = [ProviderName.OLLAMA]
    if is_ollama_configured(active_settings):
        assert active_settings.ollama_model is not None
        return RoutingDecision(
            selected_provider=ProviderName.OLLAMA,
            model=active_settings.ollama_model,
            reason=primary_reason,
            is_fallback=False,
            fallback_reason=None,
            evaluated_providers=evaluated,
        )

    if not request.allow_fallback:
        raise RoutingPolicyError(
            "Preferred local provider 'ollama' is not configured "
            "(missing OLLAMA_MODEL) and fallback is disabled."
        )

    # Fallback to OpenAI allowed only if CLOUD_ALLOWED
    if request.privacy_level == PrivacyLevel.LOCAL_ONLY:
        raise RoutingPolicyError(
            "Preferred local provider 'ollama' is not configured, and fallback to cloud "
            "provider 'openai' is strictly prohibited under 'local_only' privacy."
        )

    evaluated.append(ProviderName.OPENAI)
    if is_openai_configured(active_settings):
        assert active_settings.openai_model is not None
        return RoutingDecision(
            selected_provider=ProviderName.OPENAI,
            model=active_settings.openai_model,
            reason=(
                "Preferred local provider 'ollama' is not configured; fell back to 'openai' "
                "under cloud_allowed privacy."
            ),
            is_fallback=True,
            fallback_reason="Preferred provider 'ollama' is not configured (missing OLLAMA_MODEL).",
            evaluated_providers=evaluated,
        )

    raise RoutingPolicyError(
        "Neither preferred provider 'ollama' nor fallback provider 'openai' is configured."
    )
