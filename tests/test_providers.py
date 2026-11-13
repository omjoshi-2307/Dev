"""Unit tests for DEV provider adapters."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.config import Settings
from backend.app.providers.base import ChatMessage, ProviderConfigurationError
from backend.app.providers.ollama_provider import OllamaProvider
from backend.app.providers.openai_provider import (
    DEFAULT_OPENAI_ENDPOINT,
    OpenAIProvider,
    extract_output_text,
    sanitize_error_text,
    validate_openai_endpoint,
)


# ---------------------------------------------------------------------------
# OpenAIProvider tests (HTTPX-based Responses API)
# ---------------------------------------------------------------------------


def test_openai_provider_successful_responses_api_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["content_type"] = request.headers.get("content-type")
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "resp_test_123",
                "object": "response",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Hello from OpenAI HTTPX",
                            }
                        ],
                    }
                ],
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-openai-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            result = await provider.generate(
                [ChatMessage(role="user", content="Hello world")]
            )

            assert result.content == "Hello from OpenAI HTTPX"
            assert result.provider == "openai"
            assert result.model == "gpt-4o-mini"

    asyncio.run(run())

    assert captured["url"] == DEFAULT_OPENAI_ENDPOINT
    assert captured["authorization"] == "Bearer test-openai-key"
    assert captured["content_type"] == "application/json"
    assert captured["payload"] == {
        "model": "gpt-4o-mini",
        "input": [{"role": "user", "content": "Hello world"}],
    }


def test_openai_provider_maps_system_role_to_developer() -> None:
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload["data"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Understood"}
                        ],
                    }
                ]
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-openai-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate(
                [
                    ChatMessage(
                        role="system", content="You are a helpful assistant."
                    ),
                    ChatMessage(role="user", content="Hi"),
                ]
            )

    asyncio.run(run())

    assert captured_payload["data"]["input"] == [
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
    ]


def test_openai_provider_requires_api_key() -> None:
    provider = OpenAIProvider(
        settings=Settings(OPENAI_MODEL="gpt-4o-mini", OPENAI_API_KEY=None)
    )

    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        asyncio.run(provider.generate([ChatMessage(role="user", content="Hi")]))


def test_openai_provider_requires_model() -> None:
    provider = OpenAIProvider(
        settings=Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL=None)
    )

    with pytest.raises(ProviderConfigurationError, match="OPENAI_MODEL"):
        asyncio.run(provider.generate([ChatMessage(role="user", content="Hi")]))


def test_openai_provider_handles_401_authentication_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided: test-key-12345.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-key-12345",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(
        ProviderConfigurationError,
        match="OpenAI authentication failed: Invalid or unauthorized API key.",
    ) as exc_info:
        asyncio.run(run())

    # Ensure sensitive key material is never leaked in the error message
    assert "test-key-12345" not in str(exc_info.value)


def test_openai_provider_handles_429_rate_limit_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": "You exceeded your current quota, please check your plan and billing details.",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                }
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(
        RuntimeError,
        match="OpenAI rate limit or quota exceeded: Quota or billing limit reached.",
    ):
        asyncio.run(run())


def test_openai_provider_handles_500_server_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "error": {
                    "message": "The server had an error processing your request.",
                    "type": "server_error",
                }
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(
        RuntimeError,
        match="OpenAI service error \\(500\\): Upstream server error.",
    ) as exc_info:
        asyncio.run(run())

    # Server error body is concealed to prevent leakage
    assert "The server had an error" not in str(exc_info.value)


def test_openai_provider_handles_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(TimeoutError, match="OpenAI request timed out"):
        asyncio.run(run())


def test_openai_provider_handles_connection_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(ConnectionError, match="Failed to connect to OpenAI API"):
        asyncio.run(run())


def test_openai_provider_handles_malformed_non_json_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>Bad Gateway</body></html>")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(
        ValueError, match="OpenAI returned a malformed non-JSON response"
    ):
        asyncio.run(run())


def test_openai_provider_handles_non_dict_json_payload() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "array"])

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(
        ValueError, match="OpenAI response payload must be a JSON object"
    ):
        asyncio.run(run())


def test_openai_provider_handles_missing_output_field() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "completed"})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(
        ValueError, match="OpenAI response missing expected 'output' field"
    ):
        asyncio.run(run())


def test_openai_provider_handles_empty_output_list() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"output": []})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(
        ValueError, match="OpenAI response contains no output content"
    ):
        asyncio.run(run())


def test_openai_provider_handles_empty_or_whitespace_output() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "   \n\t  "}
                        ],
                    }
                ]
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(
        ValueError, match="OpenAI response contains empty text output"
    ):
        asyncio.run(run())


def test_extract_output_text_multi_segment_and_fallback() -> None:
    multi_segment = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Part 1 "},
                    {"type": "output_text", "text": "and Part 2"},
                ],
            }
        ]
    }
    assert extract_output_text(multi_segment) == "Part 1 and Part 2"

    direct_fallback = {"output_text": "Direct fallback content"}
    assert extract_output_text(direct_fallback) == "Direct fallback content"

    with pytest.raises(
        ValueError, match="OpenAI response contains empty text output"
    ):
        extract_output_text({"output_text": ""})

    with pytest.raises(
        ValueError, match="OpenAI response 'output' field must be a list"
    ):
        extract_output_text({"output": "not-a-list"})


def test_openai_provider_client_cleanup() -> None:
    enter_called = False
    exit_called = False

    class MockAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "MockAsyncClient":
            nonlocal enter_called
            enter_called = True
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc_val: object,
            exc_tb: object,
        ) -> None:
            nonlocal exit_called
            exit_called = True

        async def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "Cleaned up"}
                            ],
                        }
                    ]
                },
            )

    settings = Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="gpt-4o-mini")
    provider = OpenAIProvider(settings=settings)

    with patch(
        "backend.app.providers.openai_provider.httpx.AsyncClient",
        MockAsyncClient,
    ):
        result = asyncio.run(
            provider.generate([ChatMessage(role="user", content="Test cleanup")])
        )

    assert result.content == "Cleaned up"
    assert enter_called is True
    assert exit_called is True


def test_openai_provider_backwards_compatibility_legacy_client() -> None:
    response = SimpleNamespace(output_text="Legacy response from OpenAI")
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=response))
    )
    settings = Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="test-model",
    )
    # The second positional argument remains the legacy SDK client slot.
    provider = OpenAIProvider(settings, client)

    result = asyncio.run(
        provider.generate([ChatMessage(role="user", content="Hello")])
    )

    assert result.content == "Legacy response from OpenAI"
    assert result.provider == "openai"
    assert result.model == "test-model"
    client.responses.create.assert_awaited_once_with(
        model="test-model",
        input=[{"role": "user", "content": "Hello"}],
    )


# ---------------------------------------------------------------------------
# PR #4 Security Review Regression Tests
# ---------------------------------------------------------------------------


def test_mocked_429_response_does_not_expose_api_key() -> None:
    test_key = "super-secret-key-999888"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": f"Rate limit exceeded for key {test_key}: please check usage limit.",
                    "type": "insufficient_quota",
                }
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY=test_key,
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run())

    error_msg = str(exc_info.value)
    # The API key must never appear in the error message
    assert test_key not in error_msg
    assert "OpenAI rate limit or quota exceeded: Quota or billing limit reached." in error_msg
    assert "Rate limit exceeded for key" not in error_msg


def test_mocked_500_response_does_not_expose_sensitive_content() -> None:
    sensitive_db_string = (
        "postgresql://admin:super_secret_password@internal-db:5432/core"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "error": {
                    "message": f"Fatal crash in database query: {sensitive_db_string}",
                    "type": "internal_error",
                }
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            settings = Settings(
                OPENAI_API_KEY="test-key",
                OPENAI_MODEL="gpt-4o-mini",
            )
            provider = OpenAIProvider(settings=settings, http_client=client)
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run())

    error_msg = str(exc_info.value)
    # Sensitive internal server error details must never be exposed
    assert sensitive_db_string not in error_msg
    assert "super_secret_password" not in error_msg
    assert "500" in error_msg
    assert "OpenAI service error (500): Upstream server error." in error_msg


def test_validate_openai_endpoint_standalone() -> None:
    with pytest.raises(ProviderConfigurationError, match="HTTPS"):
        validate_openai_endpoint("http://api.openai.com/v1/responses")

    invalid_endpoints = [
        "https://untrusted-host.com/v1/responses",
        "https://api.openai.com:8443/v1/responses",
        "https://api.openai.com/other-path",
        "https://api.openai.com/v1/responses?redirect=https://evil.example",
        "https://user:pass@api.openai.com/v1/responses",
    ]
    for endpoint in invalid_endpoints:
        with pytest.raises(ProviderConfigurationError, match="canonical endpoint"):
            validate_openai_endpoint(endpoint)

    with pytest.raises(ProviderConfigurationError, match="non-empty string"):
        validate_openai_endpoint("")

    # Valid default endpoint succeeds without raising
    validate_openai_endpoint(DEFAULT_OPENAI_ENDPOINT)


def test_openai_provider_does_not_allow_endpoint_override() -> None:
    with pytest.raises(TypeError, match="endpoint"):
        OpenAIProvider(
            settings=Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="test"),
            endpoint="https://api.openai.com:8443/v1/responses",
        )



def test_openai_provider_retries_transient_429_and_honors_retry_after() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"error": {"code": "rate_limit_exceeded"}},
            )
        return httpx.Response(
            200,
            json={"output_text": "Recovered after rate limit"},
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = OpenAIProvider(
                settings=Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="test"),
                http_client=client,
                retry_backoff=0,
            )
            result = await provider.generate(
                [ChatMessage(role="user", content="Hi")]
            )
            assert result.content == "Recovered after rate limit"

    asyncio.run(run())
    assert attempts == 2


def test_openai_provider_retries_transient_server_error() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, text="private upstream diagnostic")
        return httpx.Response(200, json={"output_text": "Recovered"})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = OpenAIProvider(
                settings=Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="test"),
                http_client=client,
                retry_backoff=0,
            )
            result = await provider.generate(
                [ChatMessage(role="user", content="Hi")]
            )
            assert result.content == "Recovered"

    asyncio.run(run())
    assert attempts == 3


def test_openai_provider_does_not_retry_permanent_quota_failure() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            429,
            json={"error": {"code": "insufficient_quota"}},
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = OpenAIProvider(
                settings=Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="test"),
                http_client=client,
                retry_backoff=0,
            )
            with pytest.raises(RuntimeError, match="Quota or billing limit reached"):
                await provider.generate(
                    [ChatMessage(role="user", content="Hi")]
                )

    asyncio.run(run())
    assert attempts == 1


def test_openai_provider_retries_transient_connection_failure() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary network issue")
        return httpx.Response(200, json={"output_text": "Connected"})

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = OpenAIProvider(
                settings=Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="test"),
                http_client=client,
                retry_backoff=0,
            )
            result = await provider.generate(
                [ChatMessage(role="user", content="Hi")]
            )
            assert result.content == "Connected"

    asyncio.run(run())
    assert attempts == 2


def test_openai_provider_does_not_reflect_untrusted_4xx_message() -> None:
    secret_diagnostic = "postgresql://admin:private-password@internal-db"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": f"Bad request: {secret_diagnostic}"}},
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = OpenAIProvider(
                settings=Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="test"),
                http_client=client,
                retry_backoff=0,
            )
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run())
    assert "HTTP 400" in str(exc_info.value)
    assert secret_diagnostic not in str(exc_info.value)
    assert "private-password" not in str(exc_info.value)

def test_sanitize_error_text_scrubs_secrets_and_bearer_tokens() -> None:
    raw_error = (
        "Request failed for sk-1234567890abcdef and key-abcdef1234567890 with "
        "Bearer secret-token-xyz123\nMore details."
    )
    sanitized = sanitize_error_text(
        raw_error, secret_to_redact="secret-token-xyz123"
    )

    assert "sk-1234567890abcdef" not in sanitized
    assert "key-abcdef1234567890" not in sanitized
    assert "secret-token-xyz123" not in sanitized
    assert "Bearer [REDACTED]" in sanitized
    assert "[REDACTED]" in sanitized
    assert "\n" not in sanitized


# ---------------------------------------------------------------------------
# OllamaProvider tests
# ---------------------------------------------------------------------------


def test_ollama_provider_sends_non_streaming_chat_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = request.content
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Hello locally"}},
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://test",
        ) as client:
            provider = OllamaProvider(
                settings=Settings(
                    OLLAMA_BASE_URL="http://test",
                    OLLAMA_MODEL="small-model",
                ),
                http_client=client,
            )
            result = await provider.generate(
                [ChatMessage(role="user", content="Hello")]
            )
            assert result.content == "Hello locally"
            assert result.provider == "ollama"
            assert result.model == "small-model"

    asyncio.run(run())
    assert captured["url"] == "http://test/api/chat"
    assert b'"stream":false' in captured["payload"]


def test_ollama_provider_requires_model() -> None:
    provider = OllamaProvider(settings=Settings(OLLAMA_MODEL=None))

    with pytest.raises(ProviderConfigurationError, match="OLLAMA_MODEL"):
        asyncio.run(provider.generate([ChatMessage(role="user", content="Hi")]))


def test_providers_reject_empty_message_lists() -> None:
    openai_provider = OpenAIProvider(
        settings=Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="test-model"),
        client=SimpleNamespace(
            responses=SimpleNamespace(create=AsyncMock())
        ),
    )
    ollama_provider = OllamaProvider(
        settings=Settings(OLLAMA_MODEL="small-model"),
        http_client=SimpleNamespace(post=AsyncMock()),
    )

    with pytest.raises(ValueError, match="At least one"):
        asyncio.run(openai_provider.generate([]))
    with pytest.raises(ValueError, match="At least one"):
        asyncio.run(ollama_provider.generate([]))
