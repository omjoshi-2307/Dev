"""Unit tests for DEV provider adapters."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.config import Settings
from backend.app.providers.base import ChatMessage, ProviderConfigurationError
from backend.app.providers.ollama_provider import (
    OllamaProvider,
    validate_injected_client_for_local_endpoint,
)
from backend.app.providers.openai_provider import (
    DEFAULT_OPENAI_ENDPOINT,
    OpenAIProvider,
    extract_output_text,
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
    with pytest.raises(ProviderConfigurationError, match="canonical endpoint"):
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

    with patch(
        "backend.app.providers.openai_provider.asyncio.sleep",
        new_callable=AsyncMock,
    ) as sleep_mock:
        asyncio.run(run())
        sleep_mock.assert_awaited_once_with(0.0)
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

def test_openai_provider_rejects_unbounded_or_invalid_retry_configuration() -> None:
    settings = Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="test")
    with pytest.raises(ValueError, match="max_retries must be between"):
        OpenAIProvider(settings=settings, max_retries=6)
    with pytest.raises(ValueError, match="max_retries must be an integer"):
        OpenAIProvider(settings=settings, max_retries=True)
    with pytest.raises(ValueError, match="timeout must be a finite number"):
        OpenAIProvider(settings=settings, timeout=float("inf"))
    with pytest.raises(ValueError, match="timeout must be a finite number"):
        OpenAIProvider(settings=settings, timeout=0)
    with pytest.raises(ValueError, match="retry_backoff must be a finite number"):
        OpenAIProvider(settings=settings, retry_backoff=float("nan"))


def test_openai_provider_suppresses_transport_exception_chaining() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sensitive transport diagnostic")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = OpenAIProvider(
                settings=Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="test"),
                http_client=client,
                max_retries=0,
            )
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(ConnectionError) as exc_info:
        asyncio.run(run())
    assert str(exc_info.value) == "Failed to connect to OpenAI API."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


def test_openai_provider_suppresses_timeout_exception_chaining() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("sensitive timeout diagnostic")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = OpenAIProvider(
                settings=Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="test"),
                http_client=client,
                max_retries=0,
            )
            await provider.generate([ChatMessage(role="user", content="Hi")])

    with pytest.raises(TimeoutError) as exc_info:
        asyncio.run(run())
    assert str(exc_info.value) == "OpenAI request timed out."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True

def test_extract_output_text_preserves_model_refusal_text() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "refusal",
                        "refusal": "I cannot help with that request.",
                    }
                ],
            }
        ]
    }
    assert extract_output_text(payload) == "I cannot help with that request."


# ---------------------------------------------------------------------------
# OllamaProvider Proxy Hardening Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ollama_provider_loopback_disables_trust_env_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.evil.corp:8080")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.evil.corp:8080")

    recorded_kwargs: dict[str, object] = {}

    class InterceptClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            recorded_kwargs.update(kwargs)

        async def __aenter__(self) -> "InterceptClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def post(
            self, url: str, json: dict[str, object], **kwargs: object
        ) -> httpx.Response:
            recorded_kwargs.update(kwargs)
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "direct reply"}},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", InterceptClient)

    provider = OllamaProvider(
        settings=Settings(
            OLLAMA_BASE_URL="http://127.0.0.1:11434",
            OLLAMA_MODEL="small-model",
        )
    )

    resp = await provider.generate([ChatMessage(role="user", content="hello")])
    assert resp.content == "direct reply"
    assert recorded_kwargs.get("trust_env") is False


@pytest.mark.anyio
async def test_ollama_provider_remote_enables_trust_env_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_kwargs: dict[str, object] = {}

    class InterceptClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            recorded_kwargs.update(kwargs)

        async def __aenter__(self) -> "InterceptClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def post(
            self, url: str, json: dict[str, object], **kwargs: object
        ) -> httpx.Response:
            recorded_kwargs.update(kwargs)
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "remote reply"}},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", InterceptClient)

    provider = OllamaProvider(
        settings=Settings(
            OLLAMA_BASE_URL="http://remote-team-ollama.internal:11434",
            OLLAMA_MODEL="small-model",
        )
    )

    resp = await provider.generate([ChatMessage(role="user", content="hello")])
    assert resp.content == "remote reply"
    assert recorded_kwargs.get("trust_env") is True


@pytest.mark.anyio
async def test_injected_client_with_proxy_mount_rejected_for_local_loopback() -> None:
    client = httpx.AsyncClient(proxy="http://proxy.evil.corp:8080", trust_env=False)
    provider = OllamaProvider(
        settings=Settings(
            OLLAMA_BASE_URL="http://127.0.0.1:11434",
            OLLAMA_MODEL="small-model",
        ),
        http_client=client,
    )

    with pytest.raises(ProviderConfigurationError, match="proxy"):
        await provider.generate([ChatMessage(role="user", content="hello")])


@pytest.mark.anyio
async def test_injected_client_with_trust_env_and_active_proxy_rejected_for_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.evil.corp:8080")
    client = httpx.AsyncClient(trust_env=True)

    provider = OllamaProvider(
        settings=Settings(
            OLLAMA_BASE_URL="http://127.0.0.1:11434",
            OLLAMA_MODEL="small-model",
        ),
        http_client=client,
    )

    with pytest.raises(ProviderConfigurationError, match="proxy"):
        await provider.generate([ChatMessage(role="user", content="hello")])


@pytest.mark.anyio
async def test_injected_client_with_trust_env_false_accepted_for_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.evil.corp:8080")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "safe loopback reply"}},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        trust_env=False,
    ) as client:
        provider = OllamaProvider(
            settings=Settings(
                OLLAMA_BASE_URL="http://127.0.0.1:11434",
                OLLAMA_MODEL="small-model",
            ),
            http_client=client,
        )

        resp = await provider.generate([ChatMessage(role="user", content="hello")])
        assert resp.content == "safe loopback reply"


@pytest.mark.anyio
async def test_injected_client_with_proxy_allowed_for_remote_destination() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "remote proxy reply"}},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        trust_env=True,
    ) as client:
        provider = OllamaProvider(
            settings=Settings(
                OLLAMA_BASE_URL="http://remote.host:11434",
                OLLAMA_MODEL="small-model",
            ),
            http_client=client,
        )

        resp = await provider.generate([ChatMessage(role="user", content="hello")])
        assert resp.content == "remote proxy reply"


@pytest.mark.anyio
async def test_ollama_provider_explicit_trust_env_true_on_loopback_still_disables_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.evil.corp:8080")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.evil.corp:8080")

    recorded_kwargs: dict[str, object] = {}

    class InterceptClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            recorded_kwargs.update(kwargs)

        async def __aenter__(self) -> "InterceptClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def post(
            self, url: str, json: dict[str, object], **kwargs: object
        ) -> httpx.Response:
            recorded_kwargs.update(kwargs)
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "direct reply"}},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", InterceptClient)

    # Even with trust_env=True explicitly configured, loopback MUST force trust_env=False
    provider = OllamaProvider(
        settings=Settings(
            OLLAMA_BASE_URL="http://127.0.0.1:11434",
            OLLAMA_MODEL="small-model",
        ),
        trust_env=True,
    )

    resp = await provider.generate([ChatMessage(role="user", content="hello")])
    assert resp.content == "direct reply"
    assert recorded_kwargs.get("trust_env") is False


# ---------------------------------------------------------------------------
# Injected Transport and Client Hardening Tests for Local Loopback Endpoints
# ---------------------------------------------------------------------------
def test_validate_injected_client_explicit_proxy_transport_rejected() -> None:
    # Explicit proxy transport must be rejected even with trust_env=False
    proxy_transport = httpx.AsyncHTTPTransport(
        proxy="http://proxy.corporate.internal:8080"
    )
    client = httpx.AsyncClient(transport=proxy_transport, trust_env=False)
    with pytest.raises(
        ProviderConfigurationError,
        match="explicit proxy transport configured",
    ):
        validate_injected_client_for_local_endpoint(client)


def test_validate_injected_client_proxy_mounts_rejected() -> None:
    client = httpx.AsyncClient(proxy="http://proxy.corporate.internal:8080")
    with pytest.raises(
        ProviderConfigurationError,
        match="proxy mounts configured",
    ):
        validate_injected_client_for_local_endpoint(client)


def test_validate_injected_client_trust_env_true_with_env_vars_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.corporate.internal:8080")
    client = httpx.AsyncClient(trust_env=True)
    with pytest.raises(
        ProviderConfigurationError,
        match="trust_env=True while proxy environment variables are active",
    ):
        validate_injected_client_for_local_endpoint(client)


def test_validate_injected_client_safe_direct_transports_accepted() -> None:
    # 1. Standard direct async client
    direct_client = httpx.AsyncClient(trust_env=False)
    validate_injected_client_for_local_endpoint(direct_client)

    # 2. In-memory MockTransport
    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200)),
        trust_env=False,
    )
    validate_injected_client_for_local_endpoint(mock_client)

    # 3. None client passes (defaults handled internally)
    validate_injected_client_for_local_endpoint(None)


def test_validate_injected_client_custom_unverifiable_transport_rejected() -> None:
    class CustomTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            return httpx.Response(200)

    client = httpx.AsyncClient(transport=CustomTransport(), trust_env=False)
    with pytest.raises(
        ProviderConfigurationError,
        match="custom or unverifiable transport",
    ):
        validate_injected_client_for_local_endpoint(client)



def test_validate_injected_client_async_http_transport_subclass_rejected() -> None:
    class ForwardingTransport(httpx.AsyncHTTPTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            raise AssertionError("custom transport must never be executed")

    client = httpx.AsyncClient(
        transport=ForwardingTransport(),
        trust_env=False,
    )
    with pytest.raises(
        ProviderConfigurationError,
        match="custom or unverifiable transport",
    ):
        validate_injected_client_for_local_endpoint(client)


def test_validate_injected_client_mock_transport_subclass_rejected() -> None:
    class CustomMockTransport(httpx.MockTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            raise AssertionError("custom transport must never be executed")

    client = httpx.AsyncClient(
        transport=CustomMockTransport(lambda _request: httpx.Response(200)),
        trust_env=False,
    )
    with pytest.raises(
        ProviderConfigurationError,
        match="custom or unverifiable transport",
    ):
        validate_injected_client_for_local_endpoint(client)

def test_validate_injected_client_non_httpx_client_rejected() -> None:
    class DummyClient:
        pass

    with pytest.raises(
        ProviderConfigurationError,
        match="not a valid or verifiable HTTPX client",
    ):
        validate_injected_client_for_local_endpoint(DummyClient())


def test_validate_injected_client_follow_redirects_true_rejected() -> None:
    client = httpx.AsyncClient(follow_redirects=True)
    with pytest.raises(
        ProviderConfigurationError,
        match="follow_redirects=True",
    ):
        validate_injected_client_for_local_endpoint(client)


@pytest.mark.anyio
async def test_ollama_provider_injected_client_with_follow_redirects_rejected() -> None:
    client = httpx.AsyncClient(follow_redirects=True)
    provider = OllamaProvider(
        settings=Settings(
            OLLAMA_BASE_URL="http://127.0.0.1:11434",
            OLLAMA_MODEL="small-model",
        ),
        http_client=client,
    )

    with pytest.raises(
        ProviderConfigurationError,
        match="follow_redirects=True",
    ):
        await provider.generate([ChatMessage(role="user", content="hello")])


@pytest.mark.anyio
@pytest.mark.parametrize("redirect_code", [301, 302, 303, 307, 308])
async def test_ollama_provider_simulated_redirect_fails_closed_without_forwarding(
    redirect_code: int,
) -> None:
    recorded_requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(str(request.url))
        return httpx.Response(
            redirect_code,
            headers={"Location": "http://evil.remote.corp:8080/api/chat"},
            request=request,
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        trust_env=False,
        follow_redirects=False,
    )
    provider = OllamaProvider(
        settings=Settings(
            OLLAMA_BASE_URL="http://127.0.0.1:11434",
            OLLAMA_MODEL="small-model",
        ),
        http_client=client,
    )

    with pytest.raises(
        ProviderConfigurationError,
        match="refusing to follow redirects for local destination",
    ):
        await provider.generate([ChatMessage(role="user", content="secret payload")])

    # Exactly one request made to local loopback; redirect target was NEVER reached
    assert len(recorded_requests) == 1
    assert recorded_requests[0] == "http://127.0.0.1:11434/api/chat"
    assert "evil.remote.corp" not in recorded_requests[0]


def test_validate_injected_client_custom_client_with_fake_transport_rejected() -> None:
    class FakeDirectClient:
        def __init__(self) -> None:
            self._transport = httpx.MockTransport(lambda req: httpx.Response(200))
            self.trust_env = False
            self.follow_redirects = False
            self._mounts: dict[str, object] = {}

    with pytest.raises(
        ProviderConfigurationError,
        match="must be an exact httpx.AsyncClient instance",
    ):
        validate_injected_client_for_local_endpoint(FakeDirectClient())


def test_validate_injected_client_async_client_subclass_rejected() -> None:
    class SubclassedAsyncClient(httpx.AsyncClient):
        async def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return httpx.Response(200)

    client = SubclassedAsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200)),
        trust_env=False,
    )
    with pytest.raises(
        ProviderConfigurationError,
        match="must be an exact httpx.AsyncClient instance",
    ):
        validate_injected_client_for_local_endpoint(client)


def test_validate_injected_client_instance_level_post_override_rejected() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200)),
        trust_env=False,
    )
    client.__dict__["post"] = lambda *args, **kwargs: None

    with pytest.raises(
        ProviderConfigurationError,
        match="overridden 'post' method",
    ):
        validate_injected_client_for_local_endpoint(client)


@pytest.mark.anyio
async def test_ollama_provider_rejects_subclassed_async_client() -> None:
    class SubclassedAsyncClient(httpx.AsyncClient):
        pass

    client = SubclassedAsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200)),
        trust_env=False,
    )
    provider = OllamaProvider(
        settings=Settings(
            OLLAMA_BASE_URL="http://127.0.0.1:11434",
            OLLAMA_MODEL="small-model",
        ),
        http_client=client,
    )

    with pytest.raises(
        ProviderConfigurationError,
        match="must be an exact httpx.AsyncClient instance",
    ):
        await provider.generate([ChatMessage(role="user", content="hello")])




