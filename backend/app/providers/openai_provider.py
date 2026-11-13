"""OpenAI Responses API adapter using httpx."""

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from typing import Any

import httpx

from backend.app.config import Settings, get_settings
from backend.app.providers.base import (
    ChatMessage,
    ProviderConfigurationError,
    ProviderResponse,
)

DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"
MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 0.5
MAX_RETRY_AFTER_SECONDS = 30.0
_PERMANENT_QUOTA_CODES = frozenset(
    {"insufficient_quota", "billing_hard_limit_reached"}
)


def validate_openai_endpoint(endpoint: str) -> None:
    """Accept only the exact canonical OpenAI Responses API endpoint."""
    if not isinstance(endpoint, str) or not endpoint:
        raise ProviderConfigurationError(
            "OpenAI endpoint must be a non-empty string."
        )
    if endpoint != DEFAULT_OPENAI_ENDPOINT:
        raise ProviderConfigurationError(
            "OpenAI endpoint must exactly match the trusted canonical endpoint."
        )


def sanitize_error_text(text: str, secret_to_redact: str | None = None) -> str:
    """Scrub common credentials from locally generated diagnostic text."""
    if not text:
        return ""

    sanitized = text
    if secret_to_redact:
        sanitized = sanitized.replace(secret_to_redact, "[REDACTED]")
    sanitized = re.sub(
        r"Bearer\s+[^\s,;]+",
        "Bearer [REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", sanitized)
    sanitized = re.sub(r"\bkey-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", sanitized)
    sanitized = " ".join(sanitized.split())
    return sanitized[:197] + "..." if len(sanitized) > 200 else sanitized


def _error_code(response: httpx.Response) -> str | None:
    """Read only a machine-readable error code; never reflect provider text."""
    try:
        data = response.json()
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code") or error.get("type")
    return code.lower() if isinstance(code, str) else None


def _is_permanent_quota_error(response: httpx.Response) -> bool:
    return _error_code(response) in _PERMANENT_QUOTA_CODES


def _extract_error_detail(response: httpx.Response) -> str:
    """Return fixed, non-provider-controlled descriptions for HTTP failures."""
    if response.status_code == 401:
        return "Authentication failed."
    if response.status_code == 429:
        if _is_permanent_quota_error(response):
            return "Quota or billing limit reached."
        return "Rate limit exceeded."
    if response.status_code >= 500:
        return "Upstream server error."
    return f"HTTP {response.status_code}"


def _retry_delay(response: httpx.Response, attempt: int, backoff: float) -> float:
    """Honor Retry-After when valid, otherwise use bounded exponential backoff."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
                return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)
            except (TypeError, ValueError, OverflowError):
                pass
    return min(backoff * (2**attempt), MAX_RETRY_AFTER_SECONDS)


def extract_output_text(data: dict[str, Any]) -> str:
    """Extract text content from an OpenAI Responses API response payload."""
    if not isinstance(data, dict):
        raise ValueError("OpenAI response payload must be a JSON object.")

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
        client: Any | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
        max_retries: int = MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self._settings = settings or get_settings()
        self._timeout = timeout
        if max_retries < 0:
            raise ValueError("max_retries must be zero or greater.")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must be zero or greater.")
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

        if client is not None and http_client is not None:
            raise ValueError("Pass either client or http_client, not both.")

        # Keep the original second positional argument compatible with the SDK.
        if client is not None and hasattr(client, "responses"):
            self._legacy_client = client
            self._http_client = None
        elif client is not None:
            self._legacy_client = None
            self._http_client = client
        else:
            self._legacy_client = None
            self._http_client = http_client

    async def _post_with_retries(
        self,
        client: Any,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(
                    DEFAULT_OPENAI_ENDPOINT,
                    headers=headers,
                    json=payload,
                )
            except httpx.TimeoutException as exc:
                if attempt < self._max_retries:
                    await asyncio.sleep(
                        min(self._retry_backoff * (2**attempt), MAX_RETRY_AFTER_SECONDS)
                    )
                    continue
                raise TimeoutError("OpenAI request timed out.") from exc
            except httpx.RequestError as exc:
                if attempt < self._max_retries:
                    await asyncio.sleep(
                        min(self._retry_backoff * (2**attempt), MAX_RETRY_AFTER_SECONDS)
                    )
                    continue
                raise ConnectionError("Failed to connect to OpenAI API.") from exc

            retryable_status = response.status_code >= 500 or (
                response.status_code == 429
                and not _is_permanent_quota_error(response)
            )
            if retryable_status and attempt < self._max_retries:
                await asyncio.sleep(
                    _retry_delay(response, attempt, self._retry_backoff)
                )
                continue
            return response

        raise RuntimeError("OpenAI request failed after retry exhaustion.")

    async def generate(
        self,
        messages: Sequence[ChatMessage],
    ) -> ProviderResponse:
        """Send conversation messages to the configured OpenAI model."""
        api_key = self._settings.openai_api_key
        model = self._settings.openai_model
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is not configured.")
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

        # Validate the fixed endpoint before creating or attaching credentials.
        validate_openai_endpoint(DEFAULT_OPENAI_ENDPOINT)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "input": input_messages}

        if self._http_client is not None:
            response = await self._post_with_retries(
                self._http_client, headers, payload
            )
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await self._post_with_retries(client, headers, payload)

        if response.status_code == 401:
            raise ProviderConfigurationError(
                "OpenAI authentication failed: Invalid or unauthorized API key."
            )
        if response.is_error:
            detail = _extract_error_detail(response)
            if response.status_code == 429:
                raise RuntimeError(f"OpenAI rate limit or quota exceeded: {detail}")
            if response.status_code >= 500:
                raise RuntimeError(
                    f"OpenAI service error ({response.status_code}): {detail}"
                )
            raise RuntimeError(
                f"OpenAI request failed ({response.status_code}): {detail}"
            )

        try:
            data = response.json()
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "OpenAI returned a malformed non-JSON response."
            ) from exc

        return ProviderResponse(
            content=extract_output_text(data),
            provider=self.name,
            model=model,
        )
