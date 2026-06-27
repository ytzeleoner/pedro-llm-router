# pedro-llm-router — Contexto para agentes

## Propósito
Cliente async para OpenRouter con failover automático entre modelos LLM, reintentos con backoff exponencial y metadata detallada de cada request. Sin dependencias de FastAPI — funciona en cualquier contexto Python async.

## API pública (`pedro_llm_router`)

```python
from pedro_llm_router import FailoverRouter, RouterConfig, ChatMessage, RouteMetadata, RouterError
```

### `RouterConfig` (Pydantic BaseModel)
| Campo | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `openrouter_api_key` | str | `""` | Clave de OpenRouter (`sk-or-...`) |
| `models` | list[str] | 3 modelos | Lista ordenada por prioridad; se itera en orden |
| `retryPerModel` | int | 3 | Reintentos por modelo antes de pasar al siguiente |
| `delayBetweenRetriesMs` | int | 1000 | Base del backoff: `delay = (base/1000) × 2^intento` |
| `timeoutMs` | int | 30000 | Timeout total por intento |
| `neverGiveUp` | bool | False | Si True, reinicia el ciclo de modelos en lugar de lanzar RouterError |

Es serializable con `.model_dump_json()` / `RouterConfig(**json.loads(...))` — úsalo para guardar/cargar en Drive.

### `FailoverRouter`
```python
router = FailoverRouter(config: RouterConfig)

async for item in router.stream(messages: list[ChatMessage], request_id: str | None = None):
    if isinstance(item, str):
        # token de texto
    else:  # RouteMetadata — único, al final
        # metadata completa
```
Lanza `RouterError` (con `.metadata` parcial) si todos los modelos fallan y `neverGiveUp=False`.

### `ChatMessage`
```python
ChatMessage(role="system"|"user"|"assistant", content: str)
```

### `RouteMetadata`
```python
metadata.succeeded: bool
metadata.winning_model: str | None
metadata.total_latency_ms: float
metadata.total_tokens: int
metadata.failover_count: int           # modelos probados - 1
metadata.attempts: list[AttemptRecord]
metadata.request_id: str
```

### `AttemptRecord`
```python
attempt.model: str
attempt.attempt_number: int
attempt.success: bool
attempt.error_type: str | None   # "http_429" | "http_5xx" | "timeout" | "network"
attempt.latency_ms: float
attempt.tokens_received: int
```

### `RouterError`
```python
except RouterError as e:
    e.metadata  # RouteMetadata parcial con todos los intentos fallidos
```

## Patrones de uso correctos

### Uso mínimo
```python
from pedro_llm_router import FailoverRouter, RouterConfig, ChatMessage

config = RouterConfig(openrouter_api_key="sk-or-...", models=["anthropic/claude-3.5-sonnet"])
router = FailoverRouter(config)

async for item in router.stream([ChatMessage(role="user", content="Hola")]):
    if isinstance(item, str):
        print(item, end="", flush=True)
    else:
        print(f"\nModelo: {item.winning_model}")
```

### Cargar config desde JSON (p.ej. desde Drive vía pedro-gdrive)
```python
import json
from pedro_llm_router import RouterConfig

raw = pedro_gdrive.read_json("llm-router")   # dict desde Drive
config = RouterConfig(**raw)
```

### Recoger respuesta completa sin streaming
```python
tokens = []
async for item in router.stream(messages):
    if isinstance(item, str):
        tokens.append(item)
    elif isinstance(item, RouteMetadata):
        metadata = item
full_text = "".join(tokens)
```

### En FastAPI como SSE
```python
from fastapi.responses import StreamingResponse

async def generate():
    async for item in router.stream(messages):
        if isinstance(item, str):
            yield f"event: token\ndata: {item}\n\n"
        else:
            yield f"event: metadata\ndata: {item.model_dump_json()}\n\n"

return StreamingResponse(generate(), media_type="text/event-stream")
```

## Errores comunes

- **`RouterError` inmediato**: la API key es incorrecta o los modelos no existen en OpenRouter. El error será `http_401` o `http_404` en `attempt.error_type`.
- **Nunca hacer `await router.stream(...)`**: es un async generator, se itera con `async for`.
- **`RouterConfig.models` vacío**: lanza `RouterError` inmediatamente sin intentos.
- **No importar desde submódulos internos**: usar siempre `from pedro_llm_router import ...`, nunca `from pedro_llm_router.router import ...` directamente.

## Constraints

- Requiere Python ≥ 3.10 (usa `X | Y` type union syntax)
- Solo funciona con OpenRouter (`https://openrouter.ai/api/v1`) — formato OpenAI-compatible
- El streaming es token a token; si el modelo no soporta streaming, los tokens llegan en batch al final
- `neverGiveUp=True` crea un loop infinito — solo usar con timeout externo

## Tests como referencia

`tests/test_router.py` — cubre el contrato completo: respuesta exitosa, failover, RouterError, serialización de config, timestamps coherentes. Usa `respx` para mockear httpx sin red.
