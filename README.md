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
    groq_api_key="gsk_...",       # solo si usas modelos 'groq:'
    orcarouter_api_key="sk-orca-...",  # solo si usas modelos 'orca:'
    models=["anthropic/claude-3.5-sonnet", "google/gemini-pro"],
    retryPerModel=3,           # reintentos antes de pasar al siguiente modelo
    delayBetweenRetriesMs=1000,  # base del backoff exponencial
    timeoutMs=30000,           # timeout por intento
    neverGiveUp=False,         # si True, cicla infinitamente entre modelos
)
```

La config es un Pydantic model — puedes serializarla con `.model_dump_json()` y cargarla con `RouterConfig(**json.loads(...))`. Esto es lo que usa la integración con pedro-config/Drive.

## Providers

El provider **no** es un campo de configuración: se deduce del prefijo del ID del modelo. Los tres hablan la API de chat completions de OpenAI, así que el cliente es el mismo.

| Prefijo | Provider | Ejemplo de ID | API key |
|---|---|---|---|
| _(ninguno)_ | OpenRouter | `anthropic/claude-3.5-sonnet` | `openrouter_api_key` |
| `groq:` | Groq | `groq:llama-3.3-70b` | `groq_api_key` |
| `orca:` | OrcaRouter | `orca:qwen/qwen3.8-27b-free` | `orcarouter_api_key` |

El prefijo se quita antes de enviar la petición: `orca:qwen/qwen3.8-27b-free` viaja como `model: "qwen/qwen3.8-27b-free"`. Para añadir un provider nuevo basta con una entrada en `_PROVIDERS` (en `router.py`) y su campo de key en `RouterConfig`.

La lista de modelos la mantiene `pedro-config` (`check_free_models.py`), que descubre los gratuitos de los tres providers y los publica en `llm-router.json` en Drive.

## Algoritmo de failover

```
Para cada modelo en orden:
  Para cada intento (hasta retryPerModel):
    delay = (delayBetweenRetriesMs/1000) × 2^(intento-1)   ← backoff exponencial
    → si HTTP 5xx/timeout/network error: retry
    → si HTTP 429: ver "Rate limits" más abajo
    → si HTTP 4xx no reintentable: pasar al siguiente modelo
    → si éxito: yield tokens + RouteMetadata, terminar

Si todos los modelos fallan:
  → neverGiveUp=True: reiniciar desde el primero
  → neverGiveUp=False: lanzar RouterError
```

### Rate limits (429)

Un 429 no usa el backoff exponencial, sino la cabecera `Retry-After` (segundos o fecha HTTP):

- **Con `Retry-After`** dentro de `timeoutMs`: espera ese tiempo exacto y reintenta el mismo modelo. Los tiers gratuitos recargan la ventana de golpe, no gradualmente, así que el backoff exponencial solo desperdiciaría cuota.
- **Con `Retry-After` mayor que `timeoutMs`**: pasa al siguiente modelo — bloquear minutos no compensa habiendo más modelos en la cadena.
- **Sin `Retry-After`**: pasa al siguiente modelo. El provider no dice cuándo se libera la cuota, así que reintentar a ciegas no aporta.

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
