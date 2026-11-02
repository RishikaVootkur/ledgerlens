import json

import pytest

from ledgerlens.agent import graph, tools
from ledgerlens.agent.graph import (
    AgentConfig,
    finalize,
    numbers_in,
    parse_json_answer,
    quote_supported,
    run_agent,
)
from ledgerlens.agent.tools import EvidenceItem, EvidenceRegistry
from ledgerlens.config import get_settings
from ledgerlens.llm import LLMResponse, ToolCall, provider

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


def _resp(content="", tool_calls=None):
    return LLMResponse(model="m", content=content, tool_calls=tool_calls or [], prompt_tokens=100,
                       completion_tokens=20, latency_ms=50.0, cost_usd=0.001, cached=False,
                       cache_key="k")  # fmt: skip


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EVAL_DIR", str(tmp_path / "eval"))
    get_settings.cache_clear()
    monkeypatch.setattr(tools, "_search", lambda args, config: [CHUNK])
    queue: list[LLMResponse] = []
    seen: list[dict] = []

    def fake_complete(model, messages, **kwargs):
        seen.append({"purpose": kwargs.get("purpose"), "messages": messages, **kwargs})
        resp = queue.pop(0)
        provider._record(resp)  # what the real complete() does for usage tracking
        return resp

    monkeypatch.setattr(graph, "complete", fake_complete)
    yield queue, seen
    get_settings.cache_clear()


def search_call():
    args = {"query": "Apple net sales 2025", "ticker": "AAPL"}
    return ToolCall(id="1", name="search_filings", arguments=json.dumps(args))


def answer_json(**overrides):
    body = {
        "answer": "Apple's total net sales were $416,161 million in fiscal 2025 [E1].",
        "refused": False,
        "value": 416161,
        "scale": "million",
        "unit": "USD",
        "citations": [{"ref": "E1", "quote": "Total net sales were $416,161 million in 2025"}],
    }
    body.update(overrides)
    return json.dumps(body)


def test_happy_path_cites_and_converts_units(scripted):
    queue, seen = scripted
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(answer_json())]
    steps = []
    result = run_agent("What were Apple's net sales in FY2025?", AgentConfig(),
                       on_step=lambda n, d: steps.append(n))  # fmt: skip
    assert not result.refused
    assert result.answer.endswith("[1].")
    assert result.answer_value == pytest.approx(416_161e6)
    assert result.answer_unit == "USD"
    assert result.citations[0].chunk_id == CHUNK["chunk_id"]
    assert result.citations[0].page == 21
    assert result.retrieved_chunk_ids == [CHUNK["chunk_id"]]
    assert result.steps[:4] == ["plan", "search_filings", "plan", "answer"]
    assert "verify" in result.steps and "self_check" in result.steps
    assert result.usage.llm_calls == 3
    assert {"plan", "search_filings", "answer", "verify", "self_check"} <= set(steps)
    assert seen[-1]["response_format"] == {"type": "json_object"}


def test_ungrounded_number_triggers_one_repair(scripted):
    queue, seen = scripted
    bad = answer_json(answer="Net sales were $420 billion [E1].", value=420, scale="billion")
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(bad), _resp(answer_json())]
    result = run_agent("Apple net sales FY2025?", AgentConfig())
    assert seen[-1]["purpose"] == "repair"
    assert "do not appear in the evidence" in seen[-1]["messages"][-1]["content"]
    assert result.answer_value == pytest.approx(416_161e6)
    assert "repair" not in [s for s in result.steps if s.startswith("unresolved")]


def test_fake_quote_is_dropped_and_unsupported_answer_refused(scripted):
    queue, _ = scripted
    fake = answer_json(answer="Sales rose a lot.", value=None,
                       citations=[{"ref": "E1", "quote": "sales doubled to a record"}])  # fmt: skip
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(fake), _resp(fake)]
    result = run_agent("Apple sales?", AgentConfig())
    assert result.refused
    assert result.citations == []
    assert any(s.startswith("unresolved") for s in result.steps)


def test_explicit_refusal_passes_through(scripted):
    queue, _ = scripted
    refusal = json.dumps({"answer": "The filings do not disclose Apple's FY2030 sales.",
                          "refused": True, "value": None, "citations": []})  # fmt: skip
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(refusal)]
    result = run_agent("Apple FY2030 sales?", AgentConfig())
    assert result.refused and result.answer_value is None and result.citations == []


def test_max_rounds_stops_tool_loop(scripted):
    queue, _ = scripted
    queue += [_resp(tool_calls=[search_call()]) for _ in range(2)] + [_resp(answer_json())]
    result = run_agent("q", AgentConfig(max_rounds=2))
    assert result.steps.count("search_filings") == 2


def test_user_filters_override_model_arguments(scripted, monkeypatch):
    queue, _ = scripted
    captured = []
    monkeypatch.setattr(tools, "_search", lambda args, config: captured.append(args) or [CHUNK])
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(answer_json())]
    from ledgerlens.retrieval.filters import SearchFilters

    run_agent("q", AgentConfig(), filters=SearchFilters(ticker="AAPL", fiscal_year=2025))
    assert captured[0]["fiscal_year"] == 2025


def test_xbrl_tool_can_be_disabled(scripted):
    queue, seen = scripted
    queue += [_resp("READY"), _resp(json.dumps({"answer": "x", "refused": True}))]
    run_agent("q", AgentConfig(xbrl_tool=False))
    names = {t["function"]["name"] for t in seen[0]["tools"]}
    assert "get_company_facts" not in names and "search_filings" in names


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Revenue was $416.2 billion in 2025 [E1].", [416.2e9]),
        ("up 6.4% from 2024", [6.4]),
        ("1,234 and (56) million", [1234, 56e6]),
        ("In fiscal 2025, see [E12]", []),
    ],
)
def test_numbers_in(text, expected):
    assert numbers_in(text) == pytest.approx(expected)


def test_parse_json_answer_tolerates_fences_and_prose():
    assert parse_json_answer('```json\n{"answer": "a", "refused": false}\n```')["answer"] == "a"
    assert parse_json_answer('Sure! {"answer": "b"} done')["answer"] == "b"
    assert parse_json_answer("not json")["answer"] == "not json"


def test_quote_support():
    text = "Total net sales were $416,161 million in 2025."
    assert quote_supported("total net sales were $416,161 million", text)
    assert not quote_supported("net sales doubled", text)
    assert not quote_supported("", text)


def test_finalize_renumbers_and_skips_calc_sources():
    reg = EvidenceRegistry()
    for i, kind in enumerate(["chunk", "calc", "chunk"]):
        reg.add(EvidenceItem("", kind, f"src{i}", "F", "S", i, i, f"text {i}", "h"))
    draft = {"answer": "A [E3] and B [E1] via [E2].", "refused": False, "value": 5,
             "scale": "one", "unit": "percent", "citations": []}  # fmt: skip
    out = finalize(draft, reg, [])
    assert out.answer == "A [1] and B [2] via."
    assert [c.chunk_id for c in out.citations] == ["src2", "src0"]
    assert out.answer_value == 5 and out.answer_unit == "percent"
