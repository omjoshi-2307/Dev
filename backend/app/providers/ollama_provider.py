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
    """Ensure an injected HTTP client does not route local traffic through proxies, unsafe transports, or hijacked execution hooks.

    For local loopback endpoints (such as local Ollama instances under LOCAL_ONLY),
    all network traffic must be direct to prevent private message leakage.

    Inspection details and compatibility implications:
    - Injected client must be an exact httpx.AsyncClient instance. Subclasses and custom
      client wrappers are rejected because they can override request execution or routing logic.
    - Bound `post` method must resolve directly to `httpx.AsyncClient.post` without instance-level
      or class-level method overrides.
    - HTTPX encapsulates request routing in internal transports (`client._transport`)
      and optional mounted route dispatchers (`client._mounts`).
    - Standard direct async HTTPX clients use `httpx.AsyncHTTPTransport` backed by an
      `httpcore.AsyncConnectionPool` (or connection pool without proxy attributes).
    - Proxy-configured transports in HTTPX replace this pool with `httpcore.AsyncHTTPProxy`
      or `httpcore.AsyncSOCKSProxy`, or set proxy URL metadata on the transport/pool.
    - Test transports such as `httpx.MockTransport` and `httpx.ASGITransport` execute in-memory
      and do not perform remote routing; these are verified as safe.
    - Custom, unknown, or unverifiable transports whose routing cannot be established as
      strictly direct and local are rejected to fail closed against potential data exfiltration.
    """
    if client is None:
        return

    # Injected client must be strictly the exact httpx.AsyncClient class;
    # subclasses and custom client wrappers are rejected to prevent hijacked post() execution.
    if type(client) is not httpx.AsyncClient:
        client_cls_name = getattr(client, "__class__", type(client)).__name__
        raise ProviderConfigurationError(
            f"Injected HTTP client for local Ollama provider must be an exact httpx.AsyncClient instance; "
            f"custom or subclassed client '{client_cls_name}' is not a valid or verifiable HTTPX client."
        )

    # Injected client must not have an instance-level override of post()
    if "post" in getattr(client, "__dict__", {}):
        raise ProviderConfigurationError(
            "Injected HTTP client for local Ollama provider has an overridden 'post' method, "
            "violating trusted client boundaries."
        )

    # Effective bound post method must resolve strictly to the original httpx.AsyncClient.post
    post_method = getattr(client, "post", None)
    if not callable(post_method) or getattr(post_method, "__func__", None) is not httpx.AsyncClient.post:
        raise ProviderConfigurationError(
            "Injected HTTP client 'post' method has been modified or overridden, "
            "violating trusted client boundaries."
        )

    # 1. Environment proxy leakage check
    if getattr(client, "trust_env", False) and any(k in os.environ for k in PROXY_ENV_VARS):
        raise ProviderConfigurationError(
            "Injected HTTP client for local Ollama provider has trust_env=True while proxy "
            "environment variables are active, risking proxy leakage."
        )

    # 2. Redirect configuration check
    if getattr(client, "follow_redirects", False):
        raise ProviderConfigurationError(
            "Injected HTTP client for local Ollama provider has follow_redirects=True, "
            "risking redirect-based message exfiltration."
        )

    # 3. Proxy mounts check
    if hasattr(client, "_mounts") and bool(client._mounts):
        raise ProviderConfigurationError(
            "Injected HTTP client for local Ollama provider has proxy mounts configured, "
            "violating local destination boundaries."
        )

    # 4. Transport inspection
    transport = getattr(client, "_transport", None)
    if transport is None:
        raise ProviderConfigurationError(
            "Injected HTTP client has no transport configured."
        )

    # In-memory test transports with verified direct execution
    if type(transport) in (httpx.MockTransport, httpx.ASGITransport):
        return

    # Direct async HTTP transport check
    if type(transport) is httpx.AsyncHTTPTransport:
        pool = getattr(transport, "_pool", None)
        if pool is None:
            raise ProviderConfigurationError(
                "Injected HTTP client transport pool is uninitialized or missing."
            )

        pool_cls_name = pool.__class__.__name__
        if "proxy" in pool_cls_name.lower() or hasattr(pool, "_proxy_url"):
            raise ProviderConfigurationError(
                "Injected HTTP client for local Ollama provider has an explicit proxy transport "
                "configured, violating local destination boundaries."
            )

        # httpx transports can wrap custom httpcore pools. Require the exact
        # stock connection-pool class so subclasses cannot override request routing.
        if "proxy" in pool_cls_name.lower() or hasattr(pool, "_proxy_url"):
            raise ProviderConfigurationError(
                "Injected HTTP client for local Ollama provider has an explicit proxy transport "
                "configured, violating local destination boundaries."
            )

        try:
            import httpcore
        except ImportError as exc:
            raise ProviderConfigurationError(
                "Cannot verify the injected HTTP client pool without httpcore."
            ) from exc

        if type(pool) is httpcore.AsyncConnectionPool:
            return

        raise ProviderConfigurationError(
            f"Injected HTTP client transport pool '{pool_cls_name}' is custom or cannot be verified as the exact direct connection pool."
        )

    # Any custom or unknown transport class cannot be established as direct
    raise ProviderConfigurationError(
        f"Injected HTTP client uses custom or unverifiable transport '{transport.__class__.__name__}', "
        "whose routing behavior cannot be established as direct and safe."
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
                follow_redirects=False,
            )
            if is_local and 300 <= response.status_code < 400:
                raise ProviderConfigurationError(
                    f"Local Ollama endpoint responded with unexpected redirect status {response.status_code}, "
                    "refusing to follow redirects for local destination."
                )
            response.raise_for_status()
            data = response.json()
        else:
            async with httpx.AsyncClient(
                timeout=120.0,
                trust_env=effective_trust_env,
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json=payload,
                    follow_redirects=False,
                )
                if is_local and 300 <= response.status_code < 400:
                    raise ProviderConfigurationError(
                        f"Local Ollama endpoint responded with unexpected redirect status {response.status_code}, "
                        "refusing to follow redirects for local destination."
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
