# pedro-llm-router

> Cliente async para OpenRouter con failover automático entre modelos LLM y reintentos con backoff exponencial.

## ¿Qué hace?

Envías un prompt y la librería se encarga de:
1. Intentarlo con el primer modelo de tu lista (con reintentos si hay rate limit o error 5xx)
2. Si falla, pasar automáticamente al siguiente modelo
3. Devolverte los tokens en streaming + metadata completa (modelo usado, latencia, intentos)

Sin dependencias de FastAPI — funciona en cualquier script Python async, CLI o servidor web.

## Instalación

```bash
# Desde el repo (desarrollo local)
pip install -e ../pedro-llm-router

# Desde GitHub
pip install git+https://github.com/ytzeleoner/pedro-llm-router.git@master
```

## Uso rápido

```python
import asyncio
from pedro_llm_router import FailoverRouter, RouterConfig, ChatMessage

config = RouterConfig(
    openrouter_api_key="sk-or-...",
    models=[
        "anthropic/claude-3.5-sonnet",
        "mistral/mixtral-8x7b-instruct",
    ],
)
router = FailoverRouter(config)

async def main():
    messages = [ChatMessage(role="user", content="Explica los generadores async en Python")]
    async for item in router.stream(messages):
        if isinstance(item, str):
            print(item, end="", flush=True)
        else:
            print(f"\nModelo: {item.winning_model}, latencia: {item.total_latency_ms:.0f}ms")

asyncio.run(main())
```

## Funciones principales

| Símbolo | Qué hace |
|---------|----------|
| `FailoverRouter(config)` | Crea el router |
| `router.stream(messages)` | Async generator: yields tokens + RouteMetadata al final |
| `RouterConfig` | Pydantic model con toda la config (serializable a JSON) |
| `ChatMessage` | `{role: system/user/assistant, content: str}` |
| `RouteMetadata` | Resultado completo: modelo ganador, latencia, intentos |
| `RouterError` | Lanzado si todos los modelos fallan (tiene `.metadata` parcial) |

## Configuración

```python
RouterConfig(
    openrouter_api_key="sk-or-...",
    models=["anthropic/claude-3.5-sonnet", "google/gemini-pro"],
    retryPerModel=3,           # reintentos antes de pasar al siguiente modelo
    delayBetweenRetriesMs=1000,  # base del backoff exponencial
    timeoutMs=30000,           # timeout por intento
    neverGiveUp=False,         # si True, cicla infinitamente entre modelos
)
```

La config es un Pydantic model — puedes serializarla con `.model_dump_json()` y cargarla con `RouterConfig(**json.loads(...))`. Esto es lo que usa la integración con pedro-config/Drive.

## Algoritmo de failover

```
Para cada modelo en orden:
  Para cada intento (hasta retryPerModel):
    delay = (delayBetweenRetriesMs/1000) × 2^(intento-1)   ← backoff exponencial
    → si HTTP 429/5xx/timeout/network error: retry
    → si HTTP 4xx no reintentable: pasar al siguiente modelo
    → si éxito: yield tokens + RouteMetadata, terminar

Si todos los modelos fallan:
  → neverGiveUp=True: reiniciar desde el primero
  → neverGiveUp=False: lanzar RouterError
```

## Playground

Interfaz web para probar la librería en el navegador:

```bash
cd playground
pip install fastapi uvicorn
uvicorn app:app --reload
# Abre http://localhost:8000
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Integración con pedro-config

La config se puede gestionar desde Drive añadiendo `llm-router` como tipo en `pedro-config/config_types.py` (ver ese repo). El JSON en Drive tiene el mismo schema que `RouterConfig`.
