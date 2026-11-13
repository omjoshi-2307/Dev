"""Ollama local chat API adapter."""

from collections.abc import Sequence

import httpx

from backend.app.config import Settings, get_settings
from backend.app.providers.base import (
    ChatMessage,
    ProviderConfigurationError,
    ProviderResponse,
)


class OllamaProvider:
    """Generate text through a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._http_client = http_client

    @property
    def base_url(self) -> str:
        """Return the configured base URL for the Ollama server."""
        return self._settings.ollama_base_url

    async def generate(
        self,
        messages: Sequence[ChatMessage],
    ) -> ProviderResponse:
        """Send a non-streaming chat request to the local Ollama API."""

        model = self._settings.ollama_model
        if not model:
            raise ProviderConfigurationError("OLLAMA_MODEL is not configured.")
        if not messages:
            raise ValueError("At least one chat message is required.")

        base_url = self._settings.ollama_base_url.rstrip("/")
        payload = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": False,
        }

        if self._http_client is not None:
            response = await self._http_client.post(
                f"{base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        else:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Ollama returned an unexpected chat response."
            ) from exc

        if not isinstance(content, str):
            raise ValueError("Ollama response content must be a string.")

        return ProviderResponse(
            content=content,
            provider=self.name,
            model=model,
        )
