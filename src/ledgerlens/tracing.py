"""OpenTelemetry tracing into Arize Phoenix, with OpenInference span semantics.

Spans are created by hand at three choke points so every call is traced no matter who
makes it (the API, `ledgerlens ask`, the eval runner):

- `ledgerlens.agent.graph`: one AGENT span per question and one CHAIN span per graph step,
- `ledgerlens.agent.tools.run_tool`: one TOOL span per tool call (RETRIEVER for search),
- `ledgerlens.llm.provider.complete`: one LLM span per model call, with tokens, latency,
  list-price-equivalent cost and whether the response came from the cache.

Tracing is off until `setup_tracing()` succeeds. While it is off, `span()` yields a no-op
object and no attribute is computed, so the hot path costs one `None` check.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ledgerlens.logging import get_logger

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Tracer

log = get_logger(__name__)

PROJECT_NAME = "ledgerlens"
COST_LABEL = "list-price equivalent"
MAX_TEXT = 8000  # characters kept per text attribute

SpanKind = Literal["AGENT", "CHAIN", "TOOL", "LLM", "RETRIEVER"]

# OpenInference attribute names (openinference-semantic-conventions), as plain strings so
# importing this module stays cheap when tracing is off.
KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
INPUT_MIME = "input.mime_type"
OUTPUT_VALUE = "output.value"
OUTPUT_MIME = "output.mime_type"
METADATA = "metadata"

AttrValue = str | bool | int | float | Sequence[str] | Sequence[int] | Sequence[float]


class SpanLike(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_attributes(self, attributes: Mapping[str, Any]) -> None: ...


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        return None


NOOP = _NoopSpan()

_tracer: Tracer | None = None
_provider: TracerProvider | None = None


def enabled() -> bool:
    return _tracer is not None


def setup_tracing(endpoint: str | None = None, *, provider: TracerProvider | None = None) -> bool:
    """Start exporting spans. Returns True when tracing is on.

    `endpoint` defaults to the `PHOENIX_ENDPOINT` setting (for example
    http://localhost:6006/v1/traces); with neither an endpoint nor a `provider` this is a
    no-op. Tests pass their own `provider` (an SDK TracerProvider with an in-memory
    exporter). Calling it again once tracing is on does nothing.
    """
    global _tracer, _provider
    if _tracer is not None:
        return True
    if provider is None:
        from ledgerlens.config import get_settings

        settings = get_settings()
        endpoint = endpoint or settings.phoenix_endpoint
        if not endpoint:
            return False
        api_key = settings.phoenix_api_key
        try:
            from phoenix.otel import register

            provider = register(
                endpoint=endpoint,
                project_name=PROJECT_NAME,
                batch=True,
                set_global_tracer_provider=False,
                verbose=False,
                api_key=api_key.get_secret_value() if api_key else None,
                # Exit-time flushing goes through shutdown_tracing, which is time-bounded.
                shutdown_on_exit=False,
            )
        except Exception as exc:  # tracing must never stop the app from starting
            log.warning("tracing.setup_failed", error=type(exc).__name__, detail=str(exc)[:200])
            return False
        log.info("tracing.enabled", endpoint=endpoint, project=PROJECT_NAME, api_key=bool(api_key))
    _provider = provider
    _tracer = provider.get_tracer("ledgerlens")
    return True


SHUTDOWN_TIMEOUT_S = 2.0


def shutdown_tracing(timeout_s: float = SHUTDOWN_TIMEOUT_S) -> None:
    """Flush pending spans and turn tracing off (CLI exit, API shutdown, tests).

    Bounded by `timeout_s`: when the collector is down or slow, the OTLP exporter retries
    for much longer, so the flush runs in a daemon thread and is abandoned after the
    timeout instead of holding up process exit (Docker stops containers after a grace
    period)."""
    global _tracer, _provider
    provider, _provider, _tracer = _provider, None, None
    if provider is None:
        return

    def close() -> None:
        try:
            provider.force_flush(timeout_millis=int(timeout_s * 1000))
            provider.shutdown()
        except Exception as exc:
            log.warning("tracing.shutdown_failed", error=type(exc).__name__)

    worker = threading.Thread(target=close, name="tracing-shutdown", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        log.warning("tracing.shutdown_timeout", timeout_s=timeout_s)


@contextlib.contextmanager
def span(
    name: str, kind: SpanKind, attributes: Mapping[str, Any] | None = None
) -> Iterator[SpanLike]:
    """A span that is current for the block. Exceptions are recorded and re-raised."""
    if _tracer is None:
        yield NOOP
        return
    attrs = clean({KIND: kind, **(attributes or {})})
    with _tracer.start_as_current_span(name, attributes=attrs) as current:
        yield current


def annotate(attributes: Mapping[str, Any]) -> None:
    """Add attributes to the current span (no-op when tracing is off)."""
    if _tracer is None:
        return
    from opentelemetry import trace

    trace.get_current_span().set_attributes(clean(attributes))


def mark_error(message: str) -> None:
    """Flag the current span as failed for errors that are handled, not raised."""
    if _tracer is None:
        return
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    trace.get_current_span().set_status(Status(StatusCode.ERROR, message[:500]))


def clean(attributes: Mapping[str, Any]) -> dict[str, AttrValue]:
    """Drop None values and coerce the rest to types OpenTelemetry accepts."""
    out: dict[str, AttrValue] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, str):
            out[key] = value[:MAX_TEXT]
        elif isinstance(value, bool | int | float):
            out[key] = value
        elif isinstance(value, list | tuple) and all(isinstance(v, str) for v in value):
            out[key] = [str(v) for v in value]
        elif isinstance(value, list | tuple) and all(
            isinstance(v, int | float) and not isinstance(v, bool) for v in value
        ):
            out[key] = [float(v) for v in value]
        else:
            out[key] = to_json(value)
    return out


def to_json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)[:MAX_TEXT]
