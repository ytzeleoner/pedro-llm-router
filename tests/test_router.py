"""Tests del FailoverRouter con mocks de red (respx)."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest
import respx
from httpx import Response

import pedro_llm_router
from pedro_llm_router import ChatMessage, FailoverRouter, RouteMetadata, RouterConfig, RouterError
from pedro_llm_router.router import (
    GROQ_BASE,
    OPENROUTER_BASE,
    ORCAROUTER_BASE,
    _parse_retry_after,
    _resolve_provider,
)
from .conftest import sse_response


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
        # Primeros retryPerModel (2) intentos: 503; después éxito.
        # Se usa 503 y no 429 porque el 429 tiene su propia política
        # (Retry-After) y agota el modelo en un solo intento.
        if call_count <= config.retryPerModel:
            return Response(503, json={"error": "unavailable"})
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


# ── Providers con prefijo ─────────────────────────────────────────────────────

def test_resolve_provider_sin_prefijo_va_a_openrouter(config):
    base, key, real = _resolve_provider("anthropic/claude-3.5-sonnet", config)
    assert base == OPENROUTER_BASE
    assert key == "sk-or-test-key"
    assert real == "anthropic/claude-3.5-sonnet"


def test_resolve_provider_groq(config):
    base, key, real = _resolve_provider("groq:llama-3.3-70b", config)
    assert base == GROQ_BASE
    assert key == "gsk-test-key"
    assert real == "llama-3.3-70b"


def test_resolve_provider_orca(config):
    """El prefijo se quita: OrcaRouter espera el id real (vendor/model)."""
    base, key, real = _resolve_provider("orca:deepseek/deepseek-v4-flash-free", config)
    assert base == ORCAROUTER_BASE
    assert key == "sk-orca-test-key"
    assert real == "deepseek/deepseek-v4-flash-free"


@pytest.mark.asyncio
async def test_orca_usa_host_key_y_modelo_correctos(config, mock_orcarouter):
    """Un modelo 'orca:' va a api.orcarouter.ai con su key y sin el prefijo."""
    route = mock_orcarouter.post("/v1/chat/completions").mock(
        return_value=Response(200, content=sse_response(["Hola"]))
    )
    config.models = ["orca:qwen/qwen3.8-27b-free"]

    router = FailoverRouter(config)
    tokens = [
        item async for item in router.stream([ChatMessage(role="user", content="hi")])
        if isinstance(item, str)
    ]

    assert "".join(tokens) == "Hola"
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer sk-orca-test-key"
    assert json.loads(request.content)["model"] == "qwen/qwen3.8-27b-free"


@pytest.mark.asyncio
async def test_orca_no_manda_cabeceras_de_openrouter(config, mock_orcarouter):
    """HTTP-Referer / X-Title son específicas de OpenRouter."""
    route = mock_orcarouter.post("/v1/chat/completions").mock(
        return_value=Response(200, content=sse_response(["x"]))
    )
    config.models = ["orca:tencent/hy3-free"]

    router = FailoverRouter(config)
    async for _ in router.stream([ChatMessage(role="user", content="hi")]):
        pass

    headers = route.calls[0].request.headers
    assert "HTTP-Referer" not in headers
    assert "X-Title" not in headers


@pytest.mark.asyncio
async def test_failover_de_orca_a_openrouter(config, mock_all_providers):
    """Si OrcaRouter falla, la cadena sigue en un modelo de OpenRouter."""
    mock_all_providers.post("https://api.orcarouter.ai/v1/chat/completions").mock(
        return_value=Response(503, json={"error": "unavailable"})
    )
    mock_all_providers.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(200, content=sse_response(["ok"]))
    )
    config.models = ["orca:tencent/hy3-free", "anthropic/claude-3.5-sonnet"]

    router = FailoverRouter(config)
    metadata: RouteMetadata | None = None
    tokens: list[str] = []
    async for item in router.stream([ChatMessage(role="user", content="hi")]):
        if isinstance(item, str):
            tokens.append(item)
        else:
            metadata = item

    assert "".join(tokens) == "ok"
    assert metadata is not None
    assert metadata.succeeded is True
    assert metadata.winning_model == "anthropic/claude-3.5-sonnet"
    assert metadata.failover_count == 1


# ── 429 y Retry-After ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw, esperado",
    [
        ("12", 12.0),
        ("0", 0.0),
        ("  7  ", 7.0),
        ("-5", 0.0),          # negativo → 0, nunca se espera hacia atrás
        (None, None),
        ("", None),
        ("mañana", None),     # texto no parseable
    ],
)
def test_parse_retry_after(raw, esperado):
    assert _parse_retry_after(raw) == esperado


def test_parse_retry_after_fecha_http():
    """El RFC admite fecha absoluta además de segundos."""
    futuro = datetime.now(timezone.utc) + timedelta(seconds=30)
    parsed = _parse_retry_after(format_datetime(futuro, usegmt=True))
    assert parsed is not None
    assert 25 <= parsed <= 31

    pasado = datetime.now(timezone.utc) - timedelta(seconds=60)
    assert _parse_retry_after(format_datetime(pasado, usegmt=True)) == 0.0


@pytest.mark.asyncio
async def test_429_con_retry_after_reintenta_mismo_modelo(config, mock_orcarouter, monkeypatch):
    """Con Retry-After se espera ese tiempo exacto, no el backoff exponencial."""
    esperas: list[float] = []

    async def fake_sleep(delay):
        esperas.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    calls = {"n": 0}

    def side_effect(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return Response(429, headers={"Retry-After": "3"}, json={"error": "rate limit"})
        return Response(200, content=sse_response(["listo"]))

    mock_orcarouter.post("/v1/chat/completions").mock(side_effect=side_effect)
    config.models = ["orca:qwen/qwen3.8-27b-free"]

    router = FailoverRouter(config)
    tokens = [
        item async for item in router.stream([ChatMessage(role="user", content="hi")])
        if isinstance(item, str)
    ]

    assert "".join(tokens) == "listo"
    assert esperas == [3.0], "debe esperar los 3s del header, no el backoff"


@pytest.mark.asyncio
async def test_429_sin_retry_after_salta_de_modelo(config, mock_all_providers):
    """Sin el header no se reintenta: el provider no dice cuándo se libera la cuota."""
    orca = mock_all_providers.post("https://api.orcarouter.ai/v1/chat/completions").mock(
        return_value=Response(429, json={"error": "rate limit"})
    )
    mock_all_providers.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(200, content=sse_response(["ok"]))
    )
    config.models = ["orca:tencent/hy3-free", "anthropic/claude-3.5-sonnet"]

    router = FailoverRouter(config)
    metadata: RouteMetadata | None = None
    async for item in router.stream([ChatMessage(role="user", content="hi")]):
        if isinstance(item, RouteMetadata):
            metadata = item

    assert orca.call_count == 1, "no debe reintentar sin Retry-After"
    assert metadata is not None
    assert metadata.winning_model == "anthropic/claude-3.5-sonnet"


@pytest.mark.asyncio
async def test_429_con_retry_after_mayor_que_timeout_salta_de_modelo(config, mock_all_providers):
    """Bloquear más que el timeout no compensa habiendo más modelos."""
    orca = mock_all_providers.post("https://api.orcarouter.ai/v1/chat/completions").mock(
        return_value=Response(429, headers={"Retry-After": "600"}, json={"error": "rate limit"})
    )
    mock_all_providers.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(200, content=sse_response(["ok"]))
    )
    config.timeoutMs = 5000  # 5s < 600s
    config.models = ["orca:tencent/hy3-free", "anthropic/claude-3.5-sonnet"]

    router = FailoverRouter(config)
    metadata: RouteMetadata | None = None
    async for item in router.stream([ChatMessage(role="user", content="hi")]):
        if isinstance(item, RouteMetadata):
            metadata = item

    assert orca.call_count == 1
    assert metadata is not None
    assert metadata.winning_model == "anthropic/claude-3.5-sonnet"


@pytest.mark.asyncio
async def test_503_mantiene_backoff_exponencial(config, mock_openrouter, monkeypatch):
    """Los errores que no son 429 conservan el backoff de siempre."""
    esperas: list[float] = []

    async def fake_sleep(delay):
        esperas.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    mock_openrouter.post("/api/v1/chat/completions").mock(
        return_value=Response(503, json={"error": "unavailable"})
    )
    config.delayBetweenRetriesMs = 1000
    config.models = ["anthropic/claude-3.5-sonnet"]

    router = FailoverRouter(config)
    with pytest.raises(RouterError):
        async for _ in router.stream([ChatMessage(role="user", content="hi")]):
            pass

    assert esperas == [1.0], "backoff exponencial intacto para 503"
