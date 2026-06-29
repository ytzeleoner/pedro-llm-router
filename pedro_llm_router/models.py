"""Pydantic schemas públicos de pedro-llm-router."""
from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class RouterConfig(BaseModel):
    """Configuración del FailoverRouter. Serializable a/desde JSON."""
    openrouter_api_key: str = ""
    models: list[str] = Field(default_factory=list)
    retryPerModel: int = 3
    delayBetweenRetriesMs: int = 1000
    timeoutMs: int = 30000
    neverGiveUp: bool = False


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AttemptRecord(BaseModel):
    model: str
    attempt_number: int
    started_at: float
    ended_at: float
    latency_ms: float
    success: bool
    error_type: str | None = None
    error_detail: str | None = None
    tokens_received: int = 0


class RouteMetadata(BaseModel):
    """Resultado completo de una request enrutada. Último yield de FailoverRouter.stream()."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt_preview: str = ""
    winning_model: str | None = None
    total_latency_ms: float = 0.0
    attempts: list[AttemptRecord] = Field(default_factory=list)
    failover_count: int = 0
    total_tokens: int = 0
    finished_at: float = Field(default_factory=time.time)
    succeeded: bool = False
