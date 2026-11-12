"""OpenAI Responses API adapter."""

from collections.abc import Sequence

from openai import AsyncOpenAI

from backend.app.config import Settings, get_settings
from backend.app.providers.base import (
    ChatMessage,
    ProviderConfigurationError,
    ProviderResponse,
)


class OpenAIProvider:
    """Generate text with the OpenAI Responses API."""

    name = "openai"

    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client

    async def generate(
        self,
        messages: Sequence[ChatMessage],
    ) -> ProviderResponse:
        """Send conversation messages to the configured OpenAI model."""

        api_key = self._settings.openai_api_key
        model = self._settings.openai_model
        if not api_key:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY is not configured."
            )
        if not model:
            raise ProviderConfigurationError("OPENAI_MODEL is not configured.")
        if not messages:
            raise ValueError("At least one chat message is required.")

        input_messages = [
            {
                "role": "developer" if message.role == "system" else message.role,
                "content": message.content,
            }
            for message in messages
        ]

        if self._client is not None:
            response = await self._client.responses.create(
                model=model,
                input=input_messages,
            )
        else:
            async with AsyncOpenAI(api_key=api_key) as client:
                response = await client.responses.create(
                    model=model,
                    input=input_messages,
                )

        return ProviderResponse(
            content=response.output_text,
            provider=self.name,
            model=model,
        )
