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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import AsyncGenerator

import httpx

from .models import AttemptRecord, ChatMessage, RouteMetadata, RouterConfig

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GROQ_BASE = "https://api.groq.com/openai/v1"
ORCAROUTER_BASE = "https://api.orcarouter.ai/v1"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Providers con prefijo: "<prefijo><model_id>" → (base_url, campo de RouterConfig
# con la api key). Un model_id sin ninguno de estos prefijos va a OpenRouter.
# Todos exponen la API de chat completions de OpenAI, así que el cliente es el mismo.
_PROVIDERS: dict[str, tuple[str, str]] = {
    "groq:": (GROQ_BASE, "groq_api_key"),
    "orca:": (ORCAROUTER_BASE, "orcarouter_api_key"),
}

_PROVIDER_PREFIXES = tuple(_PROVIDERS)


def _resolve_provider(model_id: str, config) -> tuple[str, str, str]:
    """
    Devuelve (base_url, api_key, real_model_id) segun el prefijo del model_id.
    Sin prefijo → OpenRouter. Con prefijo ('groq:', 'orca:') → ese provider.
    """
    for prefix, (base_url, key_attr) in _PROVIDERS.items():
        if model_id.startswith(prefix):
            return base_url, getattr(config, key_attr), model_id[len(prefix):]
    return OPENROUTER_BASE, config.openrouter_api_key, model_id


def _parse_retry_after(value: str | None) -> float | None:
    """
    Convierte la cabecera Retry-After a segundos.

    Admite los dos formatos del RFC: segundos ("12") y fecha HTTP
    ("Wed, 21 Oct 2026 07:28:00 GMT"). Devuelve None si falta o no se entiende.
    Un valor negativo (fecha ya pasada) se normaliza a 0.
    """
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_at is None:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


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

        Excepción al backoff: en un 429 se respeta la cabecera Retry-After. Los
        tiers gratuitos recargan la ventana de golpe (no gradualmente), así que
        esperar el tiempo exacto que indica el provider es lo correcto; el
        backoff exponencial solo desperdiciaría cuota. Si el 429 no trae
        Retry-After no se reintenta: se pasa al siguiente modelo.

        Patrón result_holder: los async generators de Python no pueden usar
        `return value` hacia el caller que itera con `async for`, así que
        comunicamos el resultado escribiendo en una lista mutable.
        """
        attempts: list[AttemptRecord] = []
        total_tokens = 0
        retry_after: float | None = None

        for attempt_num in range(self.config.retryPerModel):
            if attempt_num > 0:
                if retry_after is not None:
                    delay = retry_after
                    logger.info(
                        "Retry-After %.2fs antes del intento %d en %s", delay, attempt_num, model,
                    )
                else:
                    delay = (self.config.delayBetweenRetriesMs / 1000) * (2 ** (attempt_num - 1))
                    logger.debug("Backoff %.2fs antes del intento %d en %s", delay, attempt_num, model)
                await asyncio.sleep(delay)

            retry_after = None

            attempt_start_ts = time.time()
            error_type: str | None = None
            error_detail: str | None = None
            success = False
            attempt_tokens = 0
            abandon_model = False

            try:
                base_url, api_key, real_model = _resolve_provider(model, self.config)
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                if base_url == OPENROUTER_BASE:
                    headers["HTTP-Referer"] = "http://localhost"
                    headers["X-Title"] = "pedro-llm-router"
                timeout = httpx.Timeout(self.config.timeoutMs / 1000)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json={
                            "model": real_model,
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
                            if response.status_code == 429:
                                retry_after = _parse_retry_after(
                                    response.headers.get("Retry-After")
                                )
                                max_wait = self.config.timeoutMs / 1000
                                if retry_after is None:
                                    # Sin Retry-After, reintentar no aporta: el provider no
                                    # dice cuándo se libera la cuota.
                                    logger.warning(
                                        "429 de %s sin Retry-After — pasando al siguiente modelo",
                                        model,
                                    )
                                    abandon_model = True
                                elif retry_after > max_wait:
                                    # Bloquear más que el timeout no compensa habiendo más
                                    # modelos en la cadena.
                                    logger.warning(
                                        "429 de %s con Retry-After %.0fs > timeout %.0fs — "
                                        "pasando al siguiente modelo",
                                        model, retry_after, max_wait,
                                    )
                                    retry_after = None
                                    abandon_model = True
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

            if abandon_model:
                break

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
