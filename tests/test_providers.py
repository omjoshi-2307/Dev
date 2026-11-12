"""Unit tests for DEV provider adapters."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from backend.app.config import Settings
from backend.app.providers.base import ChatMessage, ProviderConfigurationError
from backend.app.providers.ollama_provider import OllamaProvider
from backend.app.providers.openai_provider import OpenAIProvider


def test_openai_provider_uses_responses_api() -> None:
    response = SimpleNamespace(output_text="Hello from OpenAI")
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=response))
    )
    settings = Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="test-model",
    )
    provider = OpenAIProvider(settings=settings, client=client)

    result = asyncio.run(
        provider.generate([ChatMessage(role="user", content="Hello")])
    )

    assert result.content == "Hello from OpenAI"
    assert result.provider == "openai"
    assert result.model == "test-model"
    client.responses.create.assert_awaited_once_with(
        model="test-model",
        input=[{"role": "user", "content": "Hello"}],
    )


def test_openai_provider_maps_system_role_to_developer() -> None:
    response = SimpleNamespace(output_text="Ready")
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=response))
    )
    settings = Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="test-model",
    )
    provider = OpenAIProvider(settings=settings, client=client)

    asyncio.run(
        provider.generate([ChatMessage(role="system", content="Be concise")])
    )

    client.responses.create.assert_awaited_once_with(
        model="test-model",
        input=[{"role": "developer", "content": "Be concise"}],
    )


def test_openai_provider_requires_api_key() -> None:
    provider = OpenAIProvider(settings=Settings(OPENAI_MODEL="test-model"))

    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        asyncio.run(provider.generate([ChatMessage(role="user", content="Hi")]))


def test_ollama_provider_sends_non_streaming_chat_request() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = await request.aread()
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
