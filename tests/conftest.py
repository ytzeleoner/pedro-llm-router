"""Fixtures compartidos. Usan respx para mockear httpx sin red real."""
from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from pedro_llm_router import RouterConfig


@pytest.fixture
def config() -> RouterConfig:
    return RouterConfig(
        openrouter_api_key="sk-or-test-key",
        models=["anthropic/claude-3.5-sonnet", "mistral/mixtral-8x7b-instruct"],
        retryPerModel=2,
        delayBetweenRetriesMs=0,
        timeoutMs=5000,
    )


def sse_chunk(content: str) -> str:
    """Genera una línea SSE con un delta de contenido."""
    data = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(data)}\n\n"


def sse_response(tokens: list[str]) -> bytes:
    """Construye el body completo de una respuesta SSE de OpenRouter."""
    lines = "".join(sse_chunk(t) for t in tokens)
    lines += "data: [DONE]\n\n"
    return lines.encode()


@pytest.fixture
def mock_openrouter():
    """Contexto respx que intercepta todas las llamadas a OpenRouter."""
    with respx.mock(base_url="https://openrouter.ai") as mock:
        yield mock
