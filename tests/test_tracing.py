import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from ledgerlens import tracing
from ledgerlens.agent import tools
from ledgerlens.agent.graph import run_agent
from ledgerlens.agent.tools import EvidenceRegistry, RetrievalConfig, run_tool
from ledgerlens.api.app import create_app
from ledgerlens.config import get_settings
from ledgerlens.llm import complete, get_model, provider

MODEL = "groq-qwen3.8-27b"
CHUNK = {
    "chunk_id": "AAPL-10K-2025FY:section:00042",
    "filing_id": "AAPL-10K-2025FY",
    "ticker": "AAPL",
    "company": "Apple Inc.",
    "form": "10-K",
    "fiscal_year": 2025,
    "fiscal_period": "FY",
    "doc_name": None,
    "strategy": "section",
    "section": "Item 7. MD&A > Net Sales",
    "page": 21,
    "page_end": 21,
    "text": "Total net sales were $416,161 million in 2025 compared to $391,035 million in 2024.",
    "has_table": False,
    "table_ids": [],
}


def raw_response(content="", tool_calls=(), prompt=1000, completion=50):
    calls = [
        SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(tool_calls)
    ]
    message = SimpleNamespace(content=content, tool_calls=calls or None)
    usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage
    )


@pytest.fixture
def spans(monkeypatch, tmp_path):
    """Isolated caches, no pacing, and an in-memory span exporter."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("LLM_MODE", "live")
    get_settings.cache_clear()
    monkeypatch.setattr(provider._pacer, "wait", lambda spec, est_tokens=0: None)
    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    assert tracing.setup_tracing(provider=tp)
    yield exporter
    tracing.shutdown_tracing()
    get_settings.cache_clear()


@pytest.fixture
def fake_provider(monkeypatch):
    queue: list = []

    def call(spec, request):
        return queue.pop(0), 321.0

    monkeypatch.setattr(provider, "_call_provider", call)
    return queue


def by_name(exporter):
    out: dict[str, list] = {}
    for s in exporter.get_finished_spans():
        out.setdefault(s.name, []).append(s)
    return out


# ------------------------------------------------------------------ LLM spans


def test_llm_span_has_tokens_cost_latency_and_cache_flag(spans, fake_provider):
    fake_provider.append(raw_response("Hello there", prompt=1000, completion=50))
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    first = complete(MODEL, messages, purpose="answer")
    second = complete(MODEL, messages, purpose="answer")  # served from the disk cache
    assert (first.cached, second.cached) == (False, True)

    llm = by_name(spans)["llm.answer"]
    assert len(llm) == 2
    live, cached = (s.attributes for s in llm)
    spec = get_model(MODEL)
    expected_cost = (1000 * spec.input_per_mtok + 50 * spec.output_per_mtok) / 1e6
    for attrs in (live, cached):
        assert attrs["openinference.span.kind"] == "LLM"
        assert attrs["llm.model_name"] == "qwen/qwen3.8-27b"
        assert attrs["llm.provider"] == "groq"
        assert attrs["llm.token_count.prompt"] == 1000
        assert attrs["llm.token_count.completion"] == 50
        assert attrs["llm.token_count.total"] == 1050
        assert attrs["llm.cost.total"] == pytest.approx(expected_cost)
        assert attrs["llm.cost.prompt"] + attrs["llm.cost.completion"] == pytest.approx(
            expected_cost
        )
        assert attrs["ledgerlens.cost_label"] == "list-price equivalent"
        assert attrs["ledgerlens.latency_ms"] == 321.0
        assert attrs["ledgerlens.purpose"] == "answer"
        assert attrs["llm.input_messages.1.message.content"] == "hi"
        assert "input.value" not in attrs  # the messages above already hold the prompt
        assert attrs["output.value"] == "Hello there"
    assert live["ledgerlens.cached"] is False
    assert cached["ledgerlens.cached"] is True


def test_llm_span_records_tool_calls(spans, fake_provider):
    fake_provider.append(raw_response(tool_calls=[("search_filings", {"query": "net sales"})]))
    complete(MODEL, [{"role": "user", "content": "q"}], tools=[{"type": "function"}])
    attrs = by_name(spans)["llm.generic"][0].attributes
    prefix = "llm.output_messages.0.message.tool_calls.0.tool_call.function"
    assert attrs[f"{prefix}.name"] == "search_filings"
    assert json.loads(attrs[f"{prefix}.arguments"]) == {"query": "net sales"}


def test_llm_span_records_input_tool_calls_and_provider(spans, fake_provider):
    fake_provider.append(raw_response("ok"))
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "2"},
    ]
    complete("gemini-3.5-flash-lite", messages)
    attrs = by_name(spans)["llm.generic"][0].attributes
    assert attrs["llm.provider"] == "google"
    call = "llm.input_messages.1.message.tool_calls.0.tool_call"
    assert attrs[f"{call}.id"] == "call_1"
    assert attrs[f"{call}.function.name"] == "calculate"
    assert attrs[f"{call}.function.arguments"] == '{"expression": "1+1"}'
    assert attrs["llm.input_messages.2.message.tool_call_id"] == "call_1"
    assert attrs["llm.input_messages.2.message.content"] == "2"


def test_llm_span_records_errors(spans, monkeypatch):
    def fail(spec, request):
        raise RuntimeError("provider down")

    monkeypatch.setattr(provider, "_call_provider", fail)
    with pytest.raises(RuntimeError):
        complete(MODEL, [{"role": "user", "content": "q"}])
    (span,) = spans.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR


# ------------------------------------------------------------------ tool spans


def test_search_is_a_retriever_span(spans, monkeypatch):
    monkeypatch.setattr(tools, "_search", lambda args, config: [CHUNK])
    args = {"query": "Apple net sales", "ticker": "AAPL", "fiscal_year": 2025}
    run_tool("search_filings", args, EvidenceRegistry(), RetrievalConfig(method="bm25"))
    run_tool("search_filings", args, EvidenceRegistry(), RetrievalConfig(method="bm25"))
    first, second = by_name(spans)["search_filings"]
    attrs = first.attributes
    assert attrs["openinference.span.kind"] == "RETRIEVER"
    assert attrs["input.value"] == "Apple net sales"
    assert attrs["ledgerlens.retrieval.method"] == "bm25"
    assert json.loads(attrs["ledgerlens.retrieval.filters"]) == {
        "ticker": "AAPL",
        "fiscal_year": 2025,
    }
    assert list(attrs["ledgerlens.retrieval.chunk_ids"]) == [CHUNK["chunk_id"]]
    assert attrs["retrieval.documents.0.document.id"] == CHUNK["chunk_id"]
    assert attrs["ledgerlens.cached"] is False
    assert second.attributes["ledgerlens.cached"] is True


def test_calculate_is_a_tool_span(spans):
    run_tool("calculate", {"expression": "2 * 21"}, EvidenceRegistry(), RetrievalConfig())
    run_tool("calculate", {"expression": "import os"}, EvidenceRegistry(), RetrievalConfig())
    ok, bad = by_name(spans)["calculate"]
    assert ok.attributes["openinference.span.kind"] == "TOOL"
    assert ok.attributes["tool.name"] == "calculate"
    assert "= 42" in ok.attributes["output.value"]
    assert ok.status.status_code != StatusCode.ERROR
    assert bad.status.status_code == StatusCode.ERROR


# ------------------------------------------------------------------ whole agent


def test_agent_run_has_one_span_per_step_and_tool(spans, fake_provider, monkeypatch):
    monkeypatch.setattr(tools, "_search", lambda args, config: [CHUNK])
    answer = {
        "answer": "Apple's total net sales were $416,161 million in fiscal 2025 [E1].",
        "refused": False,
        "value": 416161,
        "scale": "million",
        "unit": "USD",
        "citations": [{"ref": "E1", "quote": "Total net sales were $416,161 million in 2025"}],
    }
    fake_provider.extend(
        [
            raw_response(tool_calls=[("search_filings", {"query": "Apple net sales 2025"})]),
            raw_response("READY"),
            raw_response(json.dumps(answer)),
        ]
    )
    result = run_agent("What were Apple's net sales in fiscal 2025?")
    assert not result.refused

    finished = spans.get_finished_spans()
    names = by_name(spans)
    ids = {s.context.span_id: s for s in finished}

    def parent(s):
        return ids[s.parent.span_id].name if s.parent else None

    (agent,) = names["agent"]
    assert agent.attributes["openinference.span.kind"] == "AGENT"
    assert agent.attributes["input.value"] == "What were Apple's net sales in fiscal 2025?"
    assert agent.attributes["output.value"] == result.answer
    assert agent.attributes["ledgerlens.usage.llm_calls"] == 3
    assert parent(agent) is None

    # One CHAIN span per graph step, all children of the agent span.
    assert len(names["plan"]) == 2
    first_plan = json.loads(names["plan"][0].attributes["input.value"])
    assert first_plan["question"] == "What were Apple's net sales in fiscal 2025?"
    (tools_step,) = names["tools"]
    tool_input = json.loads(tools_step.attributes["input.value"])
    assert tool_input["tool_calls"][0]["name"] == "search_filings"
    for step in ("plan", "tools", "answer", "verify", "self_check"):
        for s in names[step]:
            assert s.attributes["openinference.span.kind"] == "CHAIN"
            assert parent(s) == "agent"
    assert "repair" not in names

    # One span per tool call, inside the tools step.
    (search,) = names["search_filings"]
    assert parent(search) == "tools"

    # One LLM span per model call, inside the step that made it.
    assert [parent(s) for s in names["llm.gather"]] == ["plan", "plan"]
    assert [parent(s) for s in names["llm.answer"]] == ["answer"]
    assert len({s.context.trace_id for s in finished}) == 1


def test_disabled_tracing_records_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.delenv("PHOENIX_ENDPOINT", raising=False)
    get_settings.cache_clear()
    try:
        tracing.shutdown_tracing()
        assert not tracing.enabled()
        with tracing.span("x", "CHAIN", {"a": 1}) as s:
            assert s is tracing.NOOP
        tracing.annotate({"a": 1})
        tracing.mark_error("ignored")
        out = run_tool("calculate", {"expression": "1 + 1"}, EvidenceRegistry(), RetrievalConfig())
        assert out.endswith("= 2")
    finally:
        get_settings.cache_clear()


def test_setup_without_endpoint_is_a_noop(monkeypatch):
    monkeypatch.delenv("PHOENIX_ENDPOINT", raising=False)
    monkeypatch.setattr(get_settings(), "phoenix_endpoint", None)
    assert tracing.setup_tracing() is False
    assert not tracing.enabled()


def test_setup_with_endpoint_registers_phoenix(monkeypatch):
    seen = {}

    def fake_register(**kwargs):
        seen.update(kwargs)
        return TracerProvider()

    monkeypatch.setattr("phoenix.otel.register", fake_register)
    monkeypatch.setenv("PHOENIX_API_KEY", "phx-test-key")
    get_settings.cache_clear()
    try:
        assert tracing.setup_tracing("http://phoenix:6006/v1/traces") is True
        assert seen["endpoint"] == "http://phoenix:6006/v1/traces"
        assert seen["project_name"] == "ledgerlens"
        assert seen["batch"] is True
        assert seen["api_key"] == "phx-test-key"
    finally:
        tracing.shutdown_tracing()
        get_settings.cache_clear()


def test_endpoint_without_api_key_is_a_noop_logged_once(monkeypatch):
    registered, logged = [], []
    monkeypatch.setattr("phoenix.otel.register", lambda **kw: registered.append(kw))
    monkeypatch.setattr(tracing, "_skip_logged", False)
    monkeypatch.setattr(
        tracing, "log", SimpleNamespace(info=lambda event, **kw: logged.append(event))
    )
    monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
    monkeypatch.setattr(get_settings(), "phoenix_api_key", None)
    assert tracing.setup_tracing("http://phoenix:6006/v1/traces") is False
    assert tracing.setup_tracing("http://phoenix:6006/v1/traces") is False
    assert not tracing.enabled()
    assert registered == []
    assert logged == ["tracing.disabled"]


def test_clean_coerces_attribute_values():
    out = tracing.clean({"none": None, "s": "x" * 9000, "n": 3, "l": ["a", "b"], "d": {"k": 1}})
    assert "none" not in out
    assert len(out["s"]) == tracing.MAX_TEXT
    assert out["n"] == 3
    assert out["l"] == ["a", "b"]
    assert out["d"] == '{"k": 1}'


# ------------------------------------------------------------------ through the API


def agent_script():
    answer = {
        "answer": "Apple's total net sales were $416,161 million in fiscal 2025 [E1].",
        "refused": False,
        "citations": [{"ref": "E1", "quote": "Total net sales were $416,161 million in 2025"}],
    }
    return [
        raw_response(tool_calls=[("search_filings", {"query": "Apple net sales 2025"})]),
        raw_response("READY"),
        raw_response(json.dumps(answer)),
    ]


def chat(client):
    resp = client.post("/api/chat", json={"question": "Apple net sales in fiscal 2025?"})
    assert resp.status_code == 200
    return resp.text


@pytest.fixture
def api_env(monkeypatch, tmp_path, fake_provider):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    monkeypatch.setattr(provider._pacer, "wait", lambda spec, est_tokens=0: None)
    monkeypatch.setattr(tools, "_search", lambda args, config: [CHUNK])
    fake_provider.extend(agent_script())
    yield
    tracing.shutdown_tracing()
    get_settings.cache_clear()


def test_chat_request_is_one_trace_with_agent_and_children(api_env):
    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.setup_tracing(provider=tp)
    with TestClient(create_app()) as client:
        body = chat(client)
    assert "event: answer" in body

    finished = exporter.get_finished_spans()
    names = by_name(exporter)
    (agent,) = names["agent"]
    assert agent.parent is None
    assert len({s.context.trace_id for s in finished}) == 1
    children = {s.name for s in finished if s.parent and s.parent.span_id == agent.context.span_id}
    assert {"plan", "tools", "answer", "verify", "self_check"} <= children
    assert len(names["llm.gather"]) == 2
    assert names["search_filings"][0].attributes["openinference.span.kind"] == "RETRIEVER"


class FailingExporter(InMemorySpanExporter):
    def export(self, spans):
        raise ConnectionError("collector unreachable")


class HangingExporter(InMemorySpanExporter):
    def export(self, spans):
        time.sleep(10)
        return super().export(spans)


def test_failing_exporter_does_not_break_requests(api_env):
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(FailingExporter()))
    tracing.setup_tracing(provider=tp)
    with TestClient(create_app()) as client:
        body = chat(client)
    assert "event: answer" in body
    assert "event: error" not in body


def test_shutdown_is_bounded_when_the_collector_hangs(api_env):
    tp = TracerProvider()
    tp.add_span_processor(BatchSpanProcessor(HangingExporter(), schedule_delay_millis=60_000))
    tracing.setup_tracing(provider=tp)
    client = TestClient(create_app())
    client.__enter__()
    assert "event: answer" in chat(client)
    started = time.monotonic()
    client.__exit__(None, None, None)  # lifespan shutdown flushes the stuck exporter
    assert time.monotonic() - started < tracing.SHUTDOWN_TIMEOUT_S + 1.5
    assert not tracing.enabled()
