"""OpenAI Responses API adapter using httpx."""

from collections.abc import Sequence
from typing import Any

import httpx

from backend.app.config import Settings, get_settings
from backend.app.providers.base import (
    ChatMessage,
    ProviderConfigurationError,
    ProviderResponse,
)

DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"


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


def _extract_error_detail(response: httpx.Response) -> str:
    """Safely extract error message details from an HTTP response without leaking secrets."""
    try:
        data = response.json()
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict) and "message" in error:
                return str(error["message"])
            if isinstance(error, str):
                return error
    except Exception:
        pass
    text = response.text.strip()
    return text if text else f"HTTP {response.status_code}"


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
            detail = _extract_error_detail(response)
            raise RuntimeError(f"OpenAI rate limit or quota exceeded: {detail}")

        if response.status_code >= 500:
            detail = _extract_error_detail(response)
            raise RuntimeError(
                f"OpenAI service error ({response.status_code}): {detail}"
            )

        if response.is_error:
            detail = _extract_error_detail(response)
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
