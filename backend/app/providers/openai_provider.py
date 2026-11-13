"""OpenAI Responses API adapter using httpx."""

from collections.abc import Sequence
import re
from typing import Any
import urllib.parse

import httpx

from backend.app.config import Settings, get_settings
from backend.app.providers.base import (
    ChatMessage,
    ProviderConfigurationError,
    ProviderResponse,
)

DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"
TRUSTED_OPENAI_HOSTS: frozenset[str] = frozenset({"api.openai.com"})


def validate_openai_endpoint(endpoint: str) -> None:
    """Validate that the target endpoint uses HTTPS and a trusted OpenAI host.

    Enforces strict HTTPS and prevents transmitting credentials to arbitrary
    or unapproved hostnames.
    """
    if not endpoint or not isinstance(endpoint, str):
        raise ProviderConfigurationError("OpenAI endpoint must be a non-empty string.")

    try:
        parsed = urllib.parse.urlsplit(endpoint.strip())
    except Exception as exc:
        raise ProviderConfigurationError(
            f"Invalid OpenAI endpoint URL: {endpoint}"
        ) from exc

    scheme = (parsed.scheme or "").lower()
    if scheme != "https":
        raise ProviderConfigurationError(
            f"Insecure endpoint scheme '{scheme}'. OpenAI endpoints must use HTTPS."
        )

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ProviderConfigurationError(
            f"Invalid OpenAI endpoint '{endpoint}': missing hostname."
        )

    if hostname not in TRUSTED_OPENAI_HOSTS:
        raise ProviderConfigurationError(
            f"Untrusted endpoint host '{hostname}'. OpenAI credentials will not be sent to untrusted hosts."
        )


def sanitize_error_text(text: str, secret_to_redact: str | None = None) -> str:
    """Scrub sensitive credentials, tokens, and control characters from error text."""
    if not text:
        return ""

    sanitized = text

    # Redact known configured secret if provided
    if secret_to_redact and secret_to_redact.strip():
        sanitized = sanitized.replace(secret_to_redact.strip(), "[REDACTED]")

    # Redact generic bearer tokens
    sanitized = re.sub(
        r"Bearer\s+[A-Za-z0-9\-._~+/]+=*",
        "Bearer [REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Redact common API key patterns (OpenAI sk-..., key-...)
    sanitized = re.sub(r"\bsk-[A-Za-z0-9\-_]{8,}\b", "[REDACTED]", sanitized)
    sanitized = re.sub(r"\bkey-[A-Za-z0-9\-_]{8,}\b", "[REDACTED]", sanitized)

    # Normalize whitespace and collapse newlines
    sanitized = " ".join(sanitized.split())

    # Limit length to prevent log flooding with verbose upstream strings
    if len(sanitized) > 200:
        sanitized = sanitized[:197] + "..."

    return sanitized


def _extract_error_detail(
    response: httpx.Response,
    secret_to_redact: str | None = None,
) -> str:
    """Safely extract error message details without including raw upstream bodies or leaking secrets."""
    # Never parse or include raw response bodies for server errors (5xx)
    if response.status_code >= 500:
        return "Upstream server error."

    # For client errors (4xx), only extract structured message field from JSON
    try:
        data = response.json()
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return sanitize_error_text(
                        message, secret_to_redact=secret_to_redact
                    )
            elif isinstance(error, str) and error.strip():
                return sanitize_error_text(
                    error, secret_to_redact=secret_to_redact
                )
    except Exception:
        pass

    # Never include raw response.text; fall back to safe HTTP status phrase
    return f"HTTP {response.status_code}"


def extract_output_text(data: dict[str, Any]) -> str:
    """Extract text content from an OpenAI Responses API response payload."""
    if not isinstance(data, dict):
        raise ValueError("OpenAI response payload must be a JSON object.")

    # 1. Standard OpenAI Responses API format:
    # "output": [
    #     {
    #         "type": "message",
    #         "content": [
    #             {"type": "output_text", "text": "..."}
    #         ]
    #     }
    # ]
    if "output" in data:
        output = data["output"]
        if not isinstance(output, list):
            raise ValueError("OpenAI response 'output' field must be a list.")
        if not output:
            raise ValueError("OpenAI response contains no output content.")

        extracted: list[str] = []
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                content_list = item.get("content")
                if isinstance(content_list, list):
                    for part in content_list:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "output_text"
                        ):
                            text = part.get("text")
                            if isinstance(text, str):
                                extracted.append(text)

        result = "".join(extracted)
        if not result.strip():
            raise ValueError("OpenAI response contains empty text output.")
        return result

    # 2. Direct output_text fallback (convenience/mock format)
    if "output_text" in data:
        output_text = data["output_text"]
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError("OpenAI response contains empty text output.")
        return output_text

    raise ValueError("OpenAI response missing expected 'output' field.")


class OpenAIProvider:
    """Generate text with the OpenAI Responses API using httpx."""

    name = "openai"

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
        client: Any | None = None,
        endpoint: str = DEFAULT_OPENAI_ENDPOINT,
        timeout: float = 120.0,
    ) -> None:
        self._settings = settings or get_settings()
        self._endpoint = endpoint
        self._timeout = timeout

        if http_client is not None:
            self._http_client = http_client
            self._legacy_client = None
        elif client is not None:
            if hasattr(client, "responses"):
                self._http_client = None
                self._legacy_client = client
            else:
                self._http_client = client
                self._legacy_client = None
        else:
            self._http_client = None
            self._legacy_client = None

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

        # Support legacy mock clients with .responses.create(...) interface
        if self._legacy_client is not None:
            response = await self._legacy_client.responses.create(
                model=model,
                input=input_messages,
            )
            content = getattr(response, "output_text", None)
            if content is None:
                raise ValueError("OpenAI response missing output text.")
            return ProviderResponse(
                content=content,
                provider=self.name,
                model=model,
            )

        # Enforce strict endpoint security before attaching credentials or sending requests
        validate_openai_endpoint(self._endpoint)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "input": input_messages,
        }

        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        self._endpoint,
                        headers=headers,
                        json=payload,
                    )
        except httpx.TimeoutException as exc:
            raise TimeoutError("OpenAI request timed out.") from exc
        except httpx.RequestError as exc:
            raise ConnectionError(
                f"Failed to connect to OpenAI API: {exc.__class__.__name__}"
            ) from exc

        # Handle HTTP error statuses
        if response.status_code == 401:
            raise ProviderConfigurationError(
                "OpenAI authentication failed: Invalid or unauthorized API key."
            )

        if response.status_code == 429:
            detail = _extract_error_detail(response, secret_to_redact=api_key)
            raise RuntimeError(f"OpenAI rate limit or quota exceeded: {detail}")

        if response.status_code >= 500:
            detail = _extract_error_detail(response, secret_to_redact=api_key)
            raise RuntimeError(
                f"OpenAI service error ({response.status_code}): {detail}"
            )

        if response.is_error:
            detail = _extract_error_detail(response, secret_to_redact=api_key)
            raise RuntimeError(
                f"OpenAI request failed ({response.status_code}): {detail}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise ValueError(
                "OpenAI returned a malformed non-JSON response."
            ) from exc

        content = extract_output_text(data)

        return ProviderResponse(
            content=content,
            provider=self.name,
            model=model,
        )
