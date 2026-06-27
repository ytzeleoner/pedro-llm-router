"""Tests del FailoverRouter con mocks de red (respx)."""
from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

import pedro_llm_router
from pedro_llm_router import ChatMessage, FailoverRouter, RouteMetadata, RouterConfig, RouterError
from tests.conftest import sse_response


# ── API pública estable ────────────────────────────────────────────────────────

def test_api_publica():
    assert hasattr(pedro_llm_router, "FailoverRouter")
    assert hasattr(pedro_llm_router, "RouterConfig")
    assert hasattr(pedro_llm_router, "ChatMessage")
    assert hasattr(pedro_llm_router, "RouteMetadata")
    assert hasattr(pedro_llm_router, "RouterError")


# ── Caso feliz: respuesta en primer modelo ────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_respuesta_exitosa(config, mock_openrouter):
    mock_openrouter.post("/api/v1/chat/completions").mock(
        return_value=Response(200, content=sse_response(["Hola", " mundo"]))
    )

    router = FailoverRouter(config)
    messages = [ChatMessage(role="user", content="Di hola")]

    tokens: list[str] = []
    metadata: RouteMetadata | None = None

    async for item in router.stream(messages):
        if isinstance(item, str):
            tokens.append(item)
        else:
            metadata = item

    assert "".join(tokens) == "Hola mundo"
    assert metadata is not None
    assert metadata.succeeded is True
    assert metadata.winning_model == "anthropic/claude-3.5-sonnet"
    assert metadata.failover_count == 0
    assert metadata.total_tokens == 2


# ── Failover: primer modelo falla, segundo responde ───────────────────────────

@pytest.mark.asyncio
async def test_failover_al_segundo_modelo(config, mock_openrouter):
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        # Primeros retryPerModel (2) intentos: 429; después éxito
        if call_count <= config.retryPerModel:
            return Response(429, json={"error": "rate limit"})
        return Response(200, content=sse_response(["ok"]))

    mock_openrouter.post("/api/v1/chat/completions").mock(side_effect=side_effect)

    router = FailoverRouter(config)
    messages = [ChatMessage(role="user", content="Hola")]

    tokens: list[str] = []
    metadata: RouteMetadata | None = None
    async for item in router.stream(messages):
        if isinstance(item, str):
            tokens.append(item)
        else:
            metadata = item

    assert "".join(tokens) == "ok"
    assert metadata is not None
    assert metadata.succeeded is True
    assert metadata.winning_model == "mistral/mixtral-8x7b-instruct"
    assert metadata.failover_count == 1
    # Debe haber intentos fallidos del primer modelo + el exitoso del segundo
    failed = [a for a in metadata.attempts if not a.success]
    assert len(failed) == config.retryPerModel


# ── RouterError cuando todos fallan ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_error_cuando_todos_fallan(config, mock_openrouter):
    mock_openrouter.post("/api/v1/chat/completions").mock(
        return_value=Response(503, json={"error": "unavailable"})
    )

    router = FailoverRouter(config)
    messages = [ChatMessage(role="user", content="Hola")]

    with pytest.raises(RouterError) as exc_info:
        async for _ in router.stream(messages):
            pass

    err = exc_info.value
    assert err.metadata is not None
    assert err.metadata.succeeded is False
    total_attempts = len(config.models) * config.retryPerModel
    assert len(err.metadata.attempts) == total_attempts


# ── RouterConfig: serialización round-trip ────────────────────────────────────

def test_router_config_round_trip():
    cfg = RouterConfig(
        openrouter_api_key="sk-or-xxx",
        models=["anthropic/claude-3.5-sonnet"],
        retryPerModel=5,
        neverGiveUp=True,
    )
    restored = RouterConfig(**json.loads(cfg.model_dump_json()))
    assert restored == cfg


# ── RouteMetadata: timestamps coherentes ─────────────────────────────────────

@pytest.mark.asyncio
async def test_metadata_timestamps_coherentes(config, mock_openrouter):
    mock_openrouter.post("/api/v1/chat/completions").mock(
        return_value=Response(200, content=sse_response(["x"]))
    )

    router = FailoverRouter(config)
    metadata: RouteMetadata | None = None
    async for item in router.stream([ChatMessage(role="user", content="test")]):
        if isinstance(item, RouteMetadata):
            metadata = item

    assert metadata is not None
    assert metadata.total_latency_ms >= 0
    for attempt in metadata.attempts:
        assert attempt.started_at <= attempt.ended_at
        assert attempt.latency_ms >= 0
