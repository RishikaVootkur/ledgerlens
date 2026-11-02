"""FastAPI backend for the LedgerLens web UI.

The HTTP contract is defined by the UI in `web/src/api/types.ts`:

- ``POST /api/chat`` streams Server-Sent Events: ``step``, ``token``, ``answer``,
  ``error`` and ``done``.
- ``GET /api/chunks/{chunk_id}`` returns a Chunk (XBRL fact ids get a synthesized one).
- ``GET /api/models``, ``GET /api/companies``, ``GET /api/health``.
- ``GET /api/data/dashboard.json`` and ``GET /api/data/runs/{run_id}.json`` serve eval output.

The agent is synchronous, so each chat runs in a worker thread with a copy of the request's
contextvars; its step callbacks are bridged onto an asyncio queue that feeds the stream.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ledgerlens import __version__
from ledgerlens.api.facts import fact_chunk, parse_fact_id
from ledgerlens.config import get_settings
from ledgerlens.llm import DEFAULT_ANSWER_MODEL, MODELS, CacheMissError
from ledgerlens.llm.provider import QuotaExhaustedError
from ledgerlens.logging import configure_logging, get_logger
from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.schemas import AgentAnswer, Chunk
from ledgerlens.tracing import setup_tracing, shutdown_tracing
from ledgerlens.universe import BY_TICKER, UNIVERSE

log = get_logger(__name__)

MAX_QUESTION_CHARS = 1000
TOKEN_CHUNK_WORDS = 3
TOKEN_DELAY_S = 0.012
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# ------------------------------------------------------------------ request models


class ChatFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str | None = None
    fiscal_year: int | None = Field(default=None, ge=1993, le=2100)
    form: Literal["10-K", "10-Q"] | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, v: object) -> object:
        if isinstance(v, str):
            v = v.strip().upper() or None
            if v is not None and v not in BY_TICKER:
                raise ValueError(f"unknown ticker {v!r}")
        return v

    @field_validator("form", mode="before")
    @classmethod
    def _blank_form(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    filters: ChatFilters = Field(default_factory=ChatFilters)
    model: str | None = None

    @field_validator("question")
    @classmethod
    def _question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is empty")
        return v

    @field_validator("model")
    @classmethod
    def _model(cls, v: str | None) -> str | None:
        if v is not None and v not in MODELS:
            raise ValueError(f"unknown model {v!r}; known: {sorted(MODELS)}")
        return v


# ------------------------------------------------------------------ agent bridge


class ChatCancelledError(Exception):
    """Raised inside the agent thread to stop a run whose client went away."""


AgentRunner = Callable[..., AgentAnswer]


def _agent_runner() -> AgentRunner:
    """Resolved at call time so tests can monkeypatch `ledgerlens.agent.graph.run_agent`."""
    from ledgerlens.agent import graph

    return graph.run_agent


def _public_error(exc: BaseException) -> str:
    """A message safe to show in the UI: no stack traces, request bodies or keys."""
    if isinstance(exc, QuotaExhaustedError):
        return "The model's daily free-tier quota is used up. Pick another model or try later."
    if isinstance(exc, CacheMissError):
        return "Replay mode has no recorded answer for this question."
    name = type(exc).__name__
    if name in {"RateLimitError", "ServiceUnavailableError", "APIConnectionError", "Timeout"}:
        return f"The model provider is unavailable right now ({name}). Try again shortly."
    if name == "AuthenticationError":
        return "The model provider rejected the API key. Check the server configuration."
    return f"The agent failed ({name}). See the server log for details."


def _token_pieces(text: str, words: int = TOKEN_CHUNK_WORDS) -> list[str]:
    """Split text into small pieces that concatenate back to exactly `text`."""
    parts = re.findall(r"\S+\s*|\s+", text)
    return ["".join(parts[i : i + words]) for i in range(0, len(parts), words)]


def _sse(event: str, data: Any) -> ServerSentEvent:
    return ServerSentEvent(data=json.dumps(data, ensure_ascii=False), event=event)


async def chat_events(
    req: ChatRequest,
    executor: ThreadPoolExecutor,
    slots: asyncio.Semaphore,
    request_id: str,
) -> AsyncIterator[ServerSentEvent]:
    from ledgerlens.agent.graph import AgentConfig

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    cancelled = threading.Event()

    def push(kind: str, payload: Any) -> None:
        # The loop may already be gone if the client left and the server shut down.
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    def on_step(name: str, detail: str) -> None:
        if cancelled.is_set():
            raise ChatCancelledError
        push("step", {"name": name, "detail": detail})

    filters = SearchFilters(
        ticker=req.filters.ticker, fiscal_year=req.filters.fiscal_year, form=req.filters.form
    )
    config = AgentConfig(model=req.model or DEFAULT_ANSWER_MODEL)
    runner = _agent_runner()

    def work() -> None:
        try:
            push("result", runner(req.question, config, filters, on_step))
        except ChatCancelledError:
            pass  # the client is gone and nothing reads the queue any more
        except BaseException as exc:
            push("error", exc)

    started = time.perf_counter()
    log.info(
        "chat.start",
        request_id=request_id,
        model=config.model,
        question_chars=len(req.question),
        filters=filters.model_dump(exclude_none=True),
    )
    try:
        if slots.locked():
            # All agent slots are busy: tell the UI instead of sending only pings.
            yield _sse("step", {"name": "queued", "detail": "waiting for a free agent slot"})
        async with slots:
            # copy_context: usage tracking and tracing live in contextvars.
            ctx = contextvars.copy_context()
            loop.run_in_executor(executor, ctx.run, work)
            while True:
                kind, payload = await queue.get()
                if kind == "step":
                    yield _sse("step", payload)
                    continue
                if kind == "error":
                    log.error(
                        "chat.error",
                        request_id=request_id,
                        error=type(payload).__name__,
                        detail=str(payload)[:300],
                    )
                    yield _sse("error", {"message": _public_error(payload)})
                elif kind == "result":
                    answer: AgentAnswer = payload
                    for piece in _token_pieces(answer.answer):
                        yield _sse("token", {"text": piece})
                        await asyncio.sleep(TOKEN_DELAY_S)
                    yield _sse("answer", answer.model_dump(mode="json"))
                    log.info(
                        "chat.done",
                        request_id=request_id,
                        refused=answer.refused,
                        citations=len(answer.citations),
                        llm_calls=answer.usage.llm_calls,
                        cost_usd=round(answer.usage.cost_usd, 6),
                        wall_ms=round((time.perf_counter() - started) * 1000),
                    )
                break
            yield _sse("done", {})
    except asyncio.CancelledError:
        log.info("chat.client_disconnected", request_id=request_id)
        raise
    finally:
        # Stops the agent at its next step if the client went away mid-run.
        cancelled.set()


# ------------------------------------------------------------------ app


def request_id_from(raw: str | None) -> str:
    """The client's x-request-id if it is short and safe to log, otherwise a new one."""
    if raw and _REQUEST_ID.match(raw):
        return raw
    return uuid.uuid4().hex[:12]


class AccessLogMiddleware:
    """Pure ASGI access log. Logs when the last body chunk is sent, so a streamed chat is
    timed end to end (BaseHTTPMiddleware would log when the headers go out). Also assigns
    the request id, exposes it as `request.state.request_id` and echoes it back."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        raw = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = request_id_from(raw)
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status = 500
        logged = False

        def emit() -> None:
            nonlocal logged
            if logged:
                return
            logged = True
            log.info(
                "http.request",
                request_id=request_id,
                method=scope.get("method"),
                path=scope.get("path"),
                status=status,
                ms=round((time.perf_counter() - started) * 1000, 1),
            )

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                extra = (b"x-request-id", request_id.encode())
                message = {**message, "headers": [*message.get("headers", []), extra]}
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                emit()

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            emit()  # client disconnects and errors are logged too


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    app.state.executor = ThreadPoolExecutor(
        max_workers=max(1, s.max_concurrent_chats), thread_name_prefix="agent"
    )
    app.state.slots = asyncio.Semaphore(max(1, s.max_concurrent_chats))
    traced = setup_tracing()
    log.info(
        "api.start",
        version=__version__,
        data_dir=str(s.data_dir),
        llm_mode=s.llm_mode,
        tracing=traced,
    )
    try:
        yield
    finally:
        app.state.executor.shutdown(wait=False, cancel_futures=True)
        if traced:
            shutdown_tracing()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="LedgerLens API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in s.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )

    app.add_middleware(AccessLogMiddleware)

    @app.post("/api/chat")
    async def chat(req: ChatRequest, request: Request) -> EventSourceResponse:
        return EventSourceResponse(
            chat_events(
                req, request.app.state.executor, request.app.state.slots, request.state.request_id
            ),
            ping=15,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/chunks/{chunk_id:path}")
    def get_chunk(chunk_id: str) -> Chunk:
        if chunk_id.startswith("xbrl:"):
            fid = parse_fact_id(chunk_id)
            chunk = fact_chunk(fid) if fid else None
        else:
            chunk = _store_chunk(chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="chunk not found")
        return chunk

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        return {
            "models": [
                {"id": m.alias, "label": m.label, "provider": m.provider} for m in MODELS.values()
            ],
            "default": DEFAULT_ANSWER_MODEL,
        }

    @app.get("/api/companies")
    def companies() -> dict[str, Any]:
        indexed = _indexed_tickers()
        listed = [c for c in UNIVERSE if c.ticker in indexed] if indexed else list(UNIVERSE)
        return {
            "companies": [{"ticker": c.ticker, "name": c.name, "sector": c.sector} for c in listed]
        }

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        settings = get_settings()
        store, qdrant, ollama = await asyncio.gather(
            asyncio.to_thread(_check_store),
            _check_http(f"{settings.qdrant_url.rstrip('/')}/readyz"),
            _check_http(f"{settings.ollama_base_url.rstrip('/')}/api/version"),
        )
        checks = {"store": store, "qdrant": qdrant, "ollama": ollama}
        ok = all(c["ok"] for c in checks.values())
        return {"status": "ok" if ok else "degraded", "version": __version__, "checks": checks}

    @app.get("/api/data/dashboard.json")
    def dashboard() -> FileResponse:
        return _json_file(get_settings().eval_dir / "reports", "dashboard.json")

    @app.get("/api/data/runs/{run_id}.json")
    def run_file(run_id: str) -> FileResponse:
        if not _RUN_ID.match(run_id) or ".." in run_id:
            raise HTTPException(status_code=404, detail="run not found")
        return _json_file(get_settings().eval_dir / "runs", f"{run_id}.json")

    return app


# ------------------------------------------------------------------ helpers


def _store_path() -> Path:
    return get_settings().data_dir / "index" / "chunks.sqlite"


def _store_chunk(chunk_id: str) -> Chunk | None:
    from ledgerlens.retrieval.retriever import get_store

    if not _store_path().exists():  # never create an empty store from a read
        return None
    return get_store().get(chunk_id)


_tickers_cache: tuple[float, frozenset[str]] = (0.0, frozenset())


def _indexed_tickers(ttl_s: float = 60.0) -> frozenset[str]:
    """Tickers with at least one indexed filing (cached briefly)."""
    global _tickers_cache
    from ledgerlens.retrieval.retriever import get_store

    stamp, tickers = _tickers_cache
    if time.monotonic() - stamp < ttl_s and tickers:
        return tickers
    if not _store_path().exists():
        return frozenset()
    try:
        tickers = frozenset(str(f["ticker"]) for f in get_store().filings())
    except Exception as exc:
        log.warning("api.store_unreadable", error=type(exc).__name__)
        return frozenset()
    _tickers_cache = (time.monotonic(), tickers)
    return tickers


def _check_store() -> dict[str, Any]:
    from ledgerlens.retrieval.retriever import get_store

    if not _store_path().exists():
        return {"ok": False, "detail": "chunk store not found; run `make ingest`"}
    try:
        count = get_store().count()
    except Exception as exc:
        return {"ok": False, "detail": type(exc).__name__}
    return {"ok": count > 0, "detail": f"{count} chunks"}


async def _check_http(url: str) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
        return {"ok": resp.status_code < 400, "detail": f"HTTP {resp.status_code}"}
    except httpx.HTTPError as exc:
        return {"ok": False, "detail": type(exc).__name__}


def _json_file(folder: Path, name: str) -> FileResponse:
    root = folder.resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-cache"})


app = create_app()
