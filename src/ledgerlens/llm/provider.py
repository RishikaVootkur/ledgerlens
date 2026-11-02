"""The single entry point for LLM calls.

Every call goes through `complete()`, which adds:
- a disk cache keyed on the full request (reruns are free and identical),
- retries with exponential backoff and jitter on rate limits and transient errors,
- client-side pacing per model so free-tier request limits are respected,
- token, latency and list-price-equivalent cost on every response, also collected by
  any active `track_usage()` scope.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ledgerlens.config import get_settings
from ledgerlens.llm.cache import CacheMissError, ResponseCache, cache_key
from ledgerlens.llm.models import GEMINI_OPENAI_BASE, ModelSpec, get_model
from ledgerlens.logging import get_logger
from ledgerlens.schemas import Usage

log = get_logger(__name__)

NAMESPACE = "llm"
# Also strips a <think> block the model never closed (truncated output).
_THINK_RE = re.compile(r"<think>.*?(?:</think>\s*|$)", re.DOTALL)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text as the model produced it


@dataclass
class LLMResponse:
    model: str
    content: str
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    cached: bool
    cache_key: str

    def usage(self) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            llm_calls=1,
            cost_usd=self.cost_usd,
            latency_ms=self.latency_ms,
        )


# ------------------------------------------------------------------ usage scopes


@dataclass
class UsageTracker:
    calls: list[LLMResponse] = field(default_factory=list)

    @property
    def total(self) -> Usage:
        total = Usage()
        for call in self.calls:
            total = total + call.usage()
        return total


_trackers: ContextVar[tuple[UsageTracker, ...]] = ContextVar("usage_trackers", default=())


@contextlib.contextmanager
def track_usage() -> Iterator[UsageTracker]:
    """Collect every LLM call made inside the block (nested scopes all see the call)."""
    tracker = UsageTracker()
    token = _trackers.set((*_trackers.get(), tracker))
    try:
        yield tracker
    finally:
        _trackers.reset(token)


# Note: contextvars do not flow into thread-pool workers by default. Callers that fan out
# with a ThreadPoolExecutor submit `contextvars.copy_context().run` so usage is tracked.


def _record(resp: LLMResponse) -> None:
    for tracker in _trackers.get():
        tracker.calls.append(resp)


# ------------------------------------------------------------------ pacing


class _Pacer:
    """Per-model client-side pacing (thread safe).

    Requests are spaced at least 60/rpm seconds apart, and when a model has a
    tokens-per-minute cap (Groq free tier: 8000) the estimated tokens sent in the last
    60 seconds are kept under it, so throughput does not depend on 429s and backoff.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next: dict[str, float] = {}
        self._window: dict[str, list[tuple[float, int]]] = {}

    def wait(self, spec: ModelSpec, est_tokens: int = 0) -> None:
        interval = 60.0 / spec.requests_per_minute
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next.get(spec.alias, now))
            if spec.tokens_per_minute:
                window = self._window.setdefault(spec.alias, [])
                budget = spec.tokens_per_minute
                # Drop entries that will have left the window by `start`.
                window[:] = [(t, n) for t, n in window if t > start - 60]
                used = sum(n for _, n in window)
                for t, n in list(window):
                    if used + est_tokens <= budget:
                        break
                    start = max(start, t + 60)
                    used -= n
                window.append((start, min(est_tokens, budget)))
            self._next[spec.alias] = start + interval
        if (delay := start - time.monotonic()) > 0:
            time.sleep(delay)


def estimate_tokens(request: dict[str, Any]) -> int:
    """Rough token estimate for pacing: ~4 characters per token plus the output budget."""
    chars = len(json.dumps(request.get("messages", []))) + len(json.dumps(request.get("tools", [])))
    return chars // 4 + int(request.get("max_tokens", 0)) // 2


_pacer = _Pacer()


# ------------------------------------------------------------------ retries


class QuotaExhaustedError(RuntimeError):
    """A per-day free-tier quota is used up. Retrying within minutes cannot help."""


class RequestTooLargeError(RuntimeError):
    """One request is larger than the provider's per-minute token cap."""


_RETRYABLE = frozenset(
    {
        "RateLimitError",
        "ServiceUnavailableError",
        "InternalServerError",
        "BadGatewayError",
        "APIConnectionError",
        "Timeout",
    }
)
_NEVER_RETRY = frozenset(
    {
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
        "UnprocessableEntityError",
        "ContextWindowExceededError",
        "ContentPolicyViolationError",
        "APIResponseValidationError",
        "JSONSchemaValidationError",
        "UnsupportedParamsError",
        "QuotaExhaustedError",
        "RequestTooLargeError",
    }
)
_DAILY_QUOTA = re.compile(r"per[ _-]?day|PerDay|RPD|TPD|daily", re.IGNORECASE)


def _is_retryable(exc: BaseException) -> bool:
    # Match by class name so this module does not import litellm's exception tree eagerly.
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & _NEVER_RETRY:
        return False
    return bool(names & _RETRYABLE)


def _retry_after(exc: BaseException | None) -> float | None:
    """Seconds the server asked us to wait, from a `retry-after` header if present."""
    headers = getattr(exc, "headers", None) or getattr(
        getattr(exc, "response", None), "headers", None
    )
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


_backoff = wait_exponential_jitter(multiplier=2, max=90)


def _wait(state: Any) -> float:
    exc = state.outcome.exception() if state.outcome else None
    if (server := _retry_after(exc)) is not None:
        return min(server + 1.0, 120.0)
    return float(_backoff(state))


def _log_retry(state: Any) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning(
        "llm.retry",
        attempt=state.attempt_number,
        error=type(exc).__name__ if exc else None,
        sleep_s=round(state.next_action.sleep, 1) if state.next_action else None,
    )


# ------------------------------------------------------------------ core


def _cache(*, replay: bool = False) -> ResponseCache:
    """Live mode reads the local cache, then the committed one. Replay mode reads only the
    committed `eval/cache`, so a replay that passes locally also passes in CI."""
    s = get_settings()
    committed = s.eval_dir / "cache"
    if replay:
        return ResponseCache(committed)
    return ResponseCache(s.cache_dir, replay_dirs=[committed])


def _provider_kwargs(spec: ModelSpec) -> dict[str, Any]:
    s = get_settings()
    match spec.provider:
        case "gemini":
            key = s.gemini_api_key.get_secret_value() if s.gemini_api_key else None
            return {"api_base": GEMINI_OPENAI_BASE, "api_key": key}
        case "groq":
            return {"api_key": s.groq_api_key.get_secret_value() if s.groq_api_key else None}
        case "ollama":
            return {"api_base": s.ollama_base_url}


def _output_tokens(usage: Any) -> int:
    """Billable output tokens. Gemini's OpenAI endpoint reports thinking tokens only in
    total_tokens, so output = total - prompt when that is larger than completion_tokens."""
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0)
    return max(completion, total - prompt)


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=_wait,
    stop=stop_after_attempt(8),
    before_sleep=_log_retry,
    reraise=True,
)
def _call_provider(spec: ModelSpec, request: dict[str, Any]) -> tuple[Any, float]:
    import litellm

    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    _pacer.wait(spec, estimate_tokens(request))
    started = time.perf_counter()
    try:
        raw = litellm.completion(
            model=spec.litellm_model, **request, **_provider_kwargs(spec), num_retries=0
        )
    except Exception as exc:
        if type(exc).__name__ == "RateLimitError" and _DAILY_QUOTA.search(str(exc)):
            raise QuotaExhaustedError(f"{spec.alias}: daily free-tier quota exhausted") from exc
        if type(exc).__name__ == "RateLimitError" and "Request too large" in str(exc):
            # A single request above the per-minute cap can never succeed by waiting.
            raise RequestTooLargeError(f"{spec.alias}: request exceeds the per-minute cap") from exc
        raise
    return raw, (time.perf_counter() - started) * 1000


def complete(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    response_format: dict[str, Any] | None = None,
    purpose: str = "generic",
) -> LLMResponse:
    """Run one chat completion through the cache, pacing and retry layers."""
    spec = get_model(model)
    request: dict[str, Any] = {
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        **spec.extra,
    }
    if tools:
        request["tools"] = tools
        if tool_choice:
            request["tool_choice"] = tool_choice
    if response_format:
        request["response_format"] = response_format

    key = cache_key(NAMESPACE, {"model": spec.litellm_model, **request})
    cache = _cache(replay=get_settings().llm_mode == "replay")
    if (hit := cache.get(NAMESPACE, key)) is not None:
        resp = _from_record(spec, hit, key)
        _record(resp)
        return resp
    if get_settings().llm_mode == "replay":
        raise CacheMissError(f"No recorded response for {purpose} call to {model} ({key[:12]})")

    raw, latency_ms = _call_provider(spec, request)
    message = raw.choices[0].message
    content = _THINK_RE.sub("", message.content or "").strip()
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
        for tc in (message.tool_calls or [])
    ]
    finish_reason = getattr(raw.choices[0], "finish_reason", None)
    prompt_tokens = int(getattr(raw.usage, "prompt_tokens", 0) or 0)
    output_tokens = _output_tokens(raw.usage)
    record = {
        "finish_reason": finish_reason,
        "model": spec.alias,
        "purpose": purpose,
        "created_at": datetime.now(UTC).isoformat(),
        "request": {"model": spec.litellm_model, **request},
        "content": content,
        "tool_calls": [tc.__dict__ for tc in tool_calls],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "latency_ms": round(latency_ms, 1),
    }
    if content or tool_calls:
        cache.put(NAMESPACE, key, record)
    else:
        # Never cache an empty answer (often a thinking budget that ate max_tokens):
        # it would replay forever, including in the CI gate.
        log.warning("llm.empty_response", model=spec.alias, finish_reason=finish_reason)
    resp = _from_record(spec, record, key, cached=False)
    log.info(
        "llm.call",
        model=spec.alias,
        purpose=purpose,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        latency_ms=round(latency_ms),
    )
    _record(resp)
    return resp


def _from_record(
    spec: ModelSpec, record: dict[str, Any], key: str, *, cached: bool = True
) -> LLMResponse:
    prompt_tokens = int(record["prompt_tokens"])
    completion_tokens = int(record["completion_tokens"])
    return LLMResponse(
        model=spec.alias,
        content=record["content"],
        tool_calls=[ToolCall(**tc) for tc in record.get("tool_calls", [])],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        # Replays report the latency measured when the response was recorded, so
        # latency metrics stay comparable between live and replayed runs.
        latency_ms=float(record["latency_ms"]),
        cost_usd=spec.cost(prompt_tokens, completion_tokens),
        cached=cached,
        cache_key=key,
    )
