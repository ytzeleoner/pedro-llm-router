"""pedro-llm-router — FailoverRouter para OpenRouter con reintentos y failover automático."""
from .models import AttemptRecord, ChatMessage, RouteMetadata, RouterConfig
from .router import FailoverRouter, RouterError

__version__ = "0.1.0"

__all__ = [
    "FailoverRouter",
    "RouterError",
    "RouterConfig",
    "ChatMessage",
    "RouteMetadata",
    "AttemptRecord",
]
