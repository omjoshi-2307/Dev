"""Ollama local chat API adapter."""

import ipaddress
import os
import urllib.parse
from collections.abc import Sequence
from typing import Any

import httpx

from backend.app.config import Settings, get_settings
from backend.app.providers.base import (
    ChatMessage,
    ProviderConfigurationError,
    ProviderResponse,
)

PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def is_loopback_url(url: str | None) -> bool:
    """Return True if url has an http/https scheme and a verified local loopback host."""
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urllib.parse.urlsplit(url.strip())
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.username or parsed.password:
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    hostname = hostname.lower()
    if hostname in ("localhost", "localhost."):
        return True

    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_loopback
    except ValueError:
        return False


def validate_injected_client_for_local_endpoint(client: Any) -> None:
    """Ensure an injected HTTP client does not route local traffic through proxies."""
    if hasattr(client, "_mounts") and bool(client._mounts):
        raise ProviderConfigurationError(
            "Injected HTTP client for local Ollama provider has proxy transports configured, "
            "violating local destination boundaries."
        )

    if getattr(client, "trust_env", False) and any(k in os.environ for k in PROXY_ENV_VARS):
        raise ProviderConfigurationError(
            "Injected HTTP client for local Ollama provider has trust_env=True while proxy "
            "environment variables are active, risking proxy leakage."
        )


class OllamaProvider:
    """Generate text through a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: Any | None = None,
        *,
        trust_env: bool | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._http_client = http_client
        self._trust_env = trust_env

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

        is_local = is_loopback_url(base_url)
        # For local loopback destinations, trust_env is unconditionally False.
        # Explicit trust_env=True must never override this for loopback destinations,
        # ensuring environment proxy variables (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY)
        # cannot intercept or leak private local requests.
        # Remote endpoints default to trust_env=True (or explicit self._trust_env)
        # to support legitimate outbound proxies under CLOUD_ALLOWED.
        effective_trust_env = False if is_local else (
            self._trust_env if self._trust_env is not None else True
        )

        if self._http_client is not None:
            if is_local:
                validate_injected_client_for_local_endpoint(self._http_client)

            response = await self._http_client.post(
                f"{base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        else:
            async with httpx.AsyncClient(
                timeout=120.0,
                trust_env=effective_trust_env,
            ) as client:
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
