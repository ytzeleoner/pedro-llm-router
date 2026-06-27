"""Playground web para pedro-llm-router. Arranca con: uvicorn app:app --reload"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pedro_llm_router import ChatMessage, FailoverRouter, RouteMetadata, RouterConfig

app = FastAPI(title="pedro-llm-router playground", version="0.1.0")

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


class ChatPayload(BaseModel):
    api_key: str
    models: list[str]
    prompt: str
    system: str = ""
    retry_per_model: int = 2
    timeout_ms: int = 30000


@app.post("/api/chat")
async def chat(payload: ChatPayload):
    """SSE endpoint: stream de tokens + metadata al final."""
    config = RouterConfig(
        openrouter_api_key=payload.api_key,
        models=payload.models,
        retryPerModel=payload.retry_per_model,
        timeoutMs=payload.timeout_ms,
        delayBetweenRetriesMs=500,
    )
    messages: list[ChatMessage] = []
    if payload.system:
        messages.append(ChatMessage(role="system", content=payload.system))
    messages.append(ChatMessage(role="user", content=payload.prompt))

    async def generate():
        from pedro_llm_router.router import RouterError
        try:
            async for item in FailoverRouter(config).stream(messages):
                if isinstance(item, str):
                    yield f"event: token\ndata: {item}\n\n"
                elif isinstance(item, RouteMetadata):
                    yield f"event: metadata\ndata: {item.model_dump_json()}\n\n"
        except RouterError as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
