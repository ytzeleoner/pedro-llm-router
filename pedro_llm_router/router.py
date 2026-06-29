"""
FailoverRouter — cliente OpenRouter con failover automático entre modelos.

Uso básico:
    from pedro_llm_router import FailoverRouter, RouterConfig, ChatMessage

    config = RouterConfig(openrouter_api_key="sk-or-...", models=["anthropic/claude-3.5-sonnet"])
    router = FailoverRouter(config)

    async for item in router.stream([ChatMessage(role="user", content="Hola")]):
        if isinstance(item, str):
            print(item, end="", flush=True)
        else:  # RouteMetadata — último yield
            print(f"\\nModelo: {item.winning_model}, latencia: {item.total_latency_ms:.0f}ms")
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

import httpx

from .models import AttemptRecord, ChatMessage, RouteMetadata, RouterConfig

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _load_models_from_gdrive() -> list[str]:
    """Lee la lista de modelos desde llm-router.json en Google Drive (pedro-gdrive)."""
    try:
        from pedro_gdrive import read_json  # type: ignore[import]
    except ImportError as e:
        raise RouterError(
            "pedro-llm-router necesita pedro-gdrive para leer modelos. "
            "Instala con: pip install pedro-llm-router[gdrive]",
            RouteMetadata(),
        ) from e

    data = read_json("llm-router")
    if data is None:
        raise RouterError(
            "No existe llm-router.json en Google Drive. "
            "Créalo desde pedro-config antes de usar el router.",
            RouteMetadata(),
        )
    models: list[str] = data.get("models", [])
    if not models:
        raise RouterError(
            "llm-router.json en Google Drive no contiene ningún modelo. "
            "Ejecuta 'check-free-models' en pedro-config para poblar la lista.",
            RouteMetadata(),
        )
    return models


class RouterError(Exception):
    """Todos los modelos fallaron y neverGiveUp=False. Incluye metadata parcial."""
    def __init__(self, message: str, metadata: RouteMetadata) -> None:
        super().__init__(message)
        self.metadata = metadata


class FailoverRouter:
    """
    Router con failover automático entre modelos OpenRouter.

    stream() es un async generator que yields:
      - str:           cada token del modelo ganador
      - RouteMetadata: exactamente uno al final, tras todos los tokens

    Si todos los modelos fallan y neverGiveUp=False, lanza RouterError.
    """

    def __init__(self, config: RouterConfig) -> None:
        self.config = config

    # ── API pública ────────────────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[ChatMessage],
        request_id: str | None = None,
    ) -> AsyncGenerator[str | RouteMetadata, None]:
        """
        Genera tokens con failover automático entre modelos.

        Yields:
            str: tokens individuales durante el streaming
            RouteMetadata: metadata completa al final (último yield)

        Raises:
            RouterError: si todos los modelos fallan y neverGiveUp=False
        """
        request_id = request_id or str(uuid.uuid4())
        started = time.monotonic()
        all_attempts: list[AttemptRecord] = []
        models = list(self.config.models) if self.config.models else _load_models_from_gdrive()
        model_index = 0
        models_tried: set[str] = set()

        while True:
            if model_index >= len(models):
                if self.config.neverGiveUp:
                    logger.warning("Todos los modelos fallaron — reiniciando ciclo (neverGiveUp=True)")
                    model_index = 0
                else:
                    metadata = self._build_metadata(
                        request_id, messages, None, started, all_attempts, 0, succeeded=False
                    )
                    raise RouterError("Todos los modelos agotados sin respuesta exitosa", metadata)

            model = models[model_index]
            models_tried.add(model)
            logger.info("Intentando modelo %s (índice %d)", model, model_index)

            result_holder: list = []

            async for item in self._try_model(model, messages, request_id, result_holder):
                yield item

            succeeded, total_tokens, attempt_records = result_holder[0]
            all_attempts.extend(attempt_records)

            if succeeded:
                metadata = self._build_metadata(
                    request_id, messages, model, started, all_attempts,
                    total_tokens, succeeded=True,
                    failover_count=len(models_tried) - 1,
                )
                yield metadata
                return

            logger.warning("Modelo %s agotado — pasando al siguiente", model)
            model_index += 1

    # ── Internos ───────────────────────────────────────────────────────────────

    async def _try_model(
        self,
        model: str,
        messages: list[ChatMessage],
        request_id: str,
        result_holder: list,
    ) -> AsyncGenerator[str, None]:
        """
        Intenta un modelo hasta retryPerModel veces con backoff exponencial.
        Yields tokens del intento exitoso.
        Escribe [succeeded, total_tokens, attempts] en result_holder al terminar.

        Patrón result_holder: los async generators de Python no pueden usar
        `return value` hacia el caller que itera con `async for`, así que
        comunicamos el resultado escribiendo en una lista mutable.
        """
        attempts: list[AttemptRecord] = []
        total_tokens = 0

        for attempt_num in range(self.config.retryPerModel):
            if attempt_num > 0:
                delay = (self.config.delayBetweenRetriesMs / 1000) * (2 ** (attempt_num - 1))
                logger.debug("Backoff %.2fs antes del intento %d en %s", delay, attempt_num, model)
                await asyncio.sleep(delay)

            attempt_start_ts = time.time()
            error_type: str | None = None
            error_detail: str | None = None
            success = False
            attempt_tokens = 0

            try:
                timeout = httpx.Timeout(self.config.timeoutMs / 1000)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{OPENROUTER_BASE}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.config.openrouter_api_key}",
                            "HTTP-Referer": "http://localhost",
                            "X-Title": "pedro-llm-router",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": [m.model_dump() for m in messages],
                            "stream": True,
                        },
                    ) as response:
                        if response.status_code in RETRYABLE_STATUS:
                            body = await response.aread()
                            error_type = f"http_{response.status_code}"
                            error_detail = body.decode(errors="replace")[:500]
                            logger.warning(
                                "HTTP %d de %s (intento %d): %s",
                                response.status_code, model, attempt_num, error_detail[:100],
                            )
                        elif response.status_code >= 400:
                            body = await response.aread()
                            error_type = f"http_{response.status_code}"
                            error_detail = body.decode(errors="replace")[:500]
                            logger.error("HTTP %d no reintentable de %s", response.status_code, model)
                            attempts.append(self._make_attempt(
                                model, attempt_num, attempt_start_ts,
                                attempt_tokens, False, error_type, error_detail,
                            ))
                            result_holder.append((False, total_tokens, attempts))
                            return
                        else:
                            async for line in response.aiter_lines():
                                if not line.startswith("data: "):
                                    continue
                                data = line[6:].strip()
                                if data == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data)
                                    delta = (
                                        chunk.get("choices", [{}])[0]
                                        .get("delta", {})
                                        .get("content", "")
                                    )
                                    if delta:
                                        yield delta
                                        attempt_tokens += 1
                                        total_tokens += 1
                                except (json.JSONDecodeError, IndexError, KeyError):
                                    continue

                            success = True

            except httpx.TimeoutException as e:
                error_type = "timeout"
                error_detail = str(e)
                logger.warning("Timeout en %s intento %d: %s", model, attempt_num, e)

            except httpx.NetworkError as e:
                error_type = "network"
                error_detail = str(e)
                logger.warning("Error de red en %s intento %d: %s", model, attempt_num, e)

            attempts.append(self._make_attempt(
                model, attempt_num, attempt_start_ts,
                attempt_tokens, success, error_type, error_detail,
            ))

            if success:
                result_holder.append((True, total_tokens, attempts))
                return

        result_holder.append((False, total_tokens, attempts))

    def _make_attempt(
        self,
        model: str,
        attempt_num: int,
        started_at: float,
        tokens: int,
        success: bool,
        error_type: str | None,
        error_detail: str | None,
    ) -> AttemptRecord:
        ended_at = time.time()
        return AttemptRecord(
            model=model,
            attempt_number=attempt_num,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=(ended_at - started_at) * 1000,
            success=success,
            error_type=error_type,
            error_detail=error_detail,
            tokens_received=tokens,
        )

    def _build_metadata(
        self,
        request_id: str,
        messages: list[ChatMessage],
        winning_model: str | None,
        started_mono: float,
        attempts: list[AttemptRecord],
        total_tokens: int,
        succeeded: bool,
        failover_count: int = 0,
    ) -> RouteMetadata:
        total_ms = (time.monotonic() - started_mono) * 1000
        prompt_preview = ""
        for msg in reversed(messages):
            if msg.role == "user":
                prompt_preview = msg.content[:100]
                break
        return RouteMetadata(
            request_id=request_id,
            prompt_preview=prompt_preview,
            winning_model=winning_model,
            total_latency_ms=total_ms,
            attempts=attempts,
            failover_count=failover_count,
            total_tokens=total_tokens,
            succeeded=succeeded,
        )
