"""Request and response schemas for the DEV AI router."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.app.router.enums import PrivacyLevel, ProviderName, TaskComplexity

ClientMessageRole = Literal["user", "assistant"]
ConcreteProvider = Literal[ProviderName.OLLAMA, ProviderName.OPENAI]

RESERVED_METADATA_KEYS = frozenset(
    {
        "role",
        "system",
        "system_prompt",
        "system_instruction",
        "system_instructions",
        "instructions",
        "privacy_level",
        "permission",
        "permissions",
        "provider",
        "preferred_provider",
        "routing",
        "allow_fallback",
        "is_fallback",
        "identity_mismatch",
        "model_mismatch",
        "provider_reported_name",
        "provider_reported_model",
    }
)


def _get_default_privacy_level() -> PrivacyLevel:
    """Retrieve default privacy level from application settings."""
    from backend.app.config import get_settings

    return get_settings().router_default_privacy_level


def _get_default_provider() -> ProviderName:
    """Retrieve default provider preference from application settings."""
    from backend.app.config import get_settings

    return get_settings().router_default_provider


def _get_default_allow_fallback() -> bool:
    """Retrieve default fallback policy from application settings."""
    from backend.app.config import get_settings

    return get_settings().router_allow_fallback


class ChatMessageInput(BaseModel):
    """A single chat message supplied by an external client."""

    role: ClientMessageRole = Field(
        ...,
        description="Role of the message sender ('user' or 'assistant'). System prompts cannot be supplied by clients.",
    )
    content: str = Field(..., description="Message text content.")

    @field_validator("role", mode="before")
    @classmethod
    def reject_system_role(cls, value: object) -> object:
        """Reject client-supplied system messages to preserve DEV control."""
        if value == "system":
            raise ValueError(
                "Client-supplied system messages are not permitted; system prompts are managed internally by DEV."
            )
        return value

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, value: str) -> str:
        """Ensure message content is not empty or purely whitespace."""
        if not value or not value.strip():
            raise ValueError("Message content cannot be empty.")
        return value


class ChatRequest(BaseModel):
    """Client request sent to the DEV AI router.

    Client-supplied metadata is untrusted and cannot override DEV system instructions,
    permissions, privacy boundaries, or authoritative routing decisions.
    """

    messages: list[ChatMessageInput] = Field(
        ...,
        min_length=1,
        description="Non-empty sequence of conversation messages.",
    )
    privacy_level: PrivacyLevel = Field(
        default_factory=_get_default_privacy_level,
        description="Privacy constraint for this request. Defaults strictly to 'local_only'.",
    )
    complexity: TaskComplexity = Field(
        default=TaskComplexity.AUTO,
        description="Declared or estimated complexity level. Defaults to 'auto'.",
    )
    preferred_provider: ProviderName = Field(
        default_factory=_get_default_provider,
        description="Explicit user override for provider. Defaults to configured router default ('auto').",
    )
    allow_fallback: bool = Field(
        default_factory=_get_default_allow_fallback,
        description="Whether fallback to alternative providers is permitted. Defaults to False.",
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description=(
            "Untrusted client-supplied metadata for correlation or tracking. "
            "Client metadata is strictly untrusted and cannot override DEV system instructions, "
            "security permissions, privacy boundaries, or authoritative routing decisions."
        ),
    )

    @field_validator("messages")
    @classmethod
    def validate_messages_non_empty(
        cls, value: list[ChatMessageInput]
    ) -> list[ChatMessageInput]:
        """Validate that at least one message is provided."""
        if not value:
            raise ValueError("At least one message is required.")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata_untrusted_boundaries(
        cls, value: dict[str, object]
    ) -> dict[str, object]:
        """Ensure client metadata does not attempt to smuggle or override system instructions, permissions, or routing."""
        lowered_keys = {str(k).lower() for k in value.keys()}
        conflicts = lowered_keys.intersection(RESERVED_METADATA_KEYS)
        if conflicts:
            raise ValueError(
                f"Client metadata contains reserved system key(s): {sorted(conflicts)}. "
                "Metadata is untrusted and cannot override system instructions, permissions, or routing decisions."
            )
        return value


class RoutingDecision(BaseModel):
    """Deterministic routing decision details produced by the AI router."""

    selected_provider: ConcreteProvider = Field(
        ...,
        description="The concrete provider selected for generation (cannot be 'auto').",
    )
    model: str = Field(..., description="The model name to execute.")
    reason: str = Field(..., description="Explanation of the routing decision.")
    is_fallback: bool = Field(
        default=False,
        description="Whether this selection represents a fallback from an initially preferred choice.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Reason for fallback if a fallback occurred.",
    )
    evaluated_providers: list[ConcreteProvider] = Field(
        default_factory=list,
        description="List of concrete providers evaluated during routing.",
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Extensible metadata dictionary for routing diagnostics.",
    )


class ChatResponse(BaseModel):
    """Normalized response returned to the client by the DEV AI router."""

    content: str = Field(..., description="Generated text content.")
    provider: ConcreteProvider = Field(
        ...,
        description="Concrete provider that generated the response (cannot be 'auto').",
    )
    model: str = Field(..., description="Model name that generated the response.")
    routing: RoutingDecision = Field(..., description="Audit details of the routing decision.")
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Extensible metadata dictionary for response metrics.",
    )
