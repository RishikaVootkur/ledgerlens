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
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
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
    result = run_agent("q", AgentConfig(max_rounds=2, dedupe_tool_calls=False))
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


# ---- review regressions (PR #8) ------------------------------------------------


def test_dropped_citation_marker_cannot_pass_self_check(scripted):
    queue, _ = scripted
    fake = answer_json(answer="Net sales doubled to a record [E1].", value=None,
                       citations=[{"ref": "E1", "quote": "net sales doubled to a record"}])  # fmt: skip
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(fake), _resp(fake)]
    result = run_agent("Apple sales?", AgentConfig())
    assert result.refused and result.citations == []
    assert "[" not in result.answer


def test_empty_quote_on_passage_is_rejected(scripted):
    queue, _ = scripted
    no_quote = answer_json(citations=[{"ref": "E1", "quote": ""}])
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(no_quote), _resp(no_quote)]
    assert run_agent("q", AgentConfig()).refused


def test_invented_number_after_repair_is_refused(scripted):
    queue, _ = scripted
    bad = answer_json(answer="Net sales were $420 billion [E1].", value=420, scale="billion")
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(bad), _resp(bad)]
    result = run_agent("q", AgentConfig())
    assert result.refused and result.answer_value is None


def test_small_percentages_are_checked(scripted):
    queue, _ = scripted
    bad = answer_json(answer="Total net sales were $416,161 million in 2025, up 27.4% [E1].")
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(bad), _resp(bad)]
    assert run_agent("q", AgentConfig()).refused


def test_calculate_cannot_launder_invented_inputs():
    from ledgerlens.agent.graph import grounding_sources, ungrounded_numbers

    reg = EvidenceRegistry()
    reg.add(
        EvidenceItem("", "chunk", "c1", "F", "S", 1, 1, "Net sales were $416,161 million.", "h")
    )
    reg.add(
        EvidenceItem(
            "", "calc", "calc:523000 * 1", None, "calc", None, None, "523000 * 1 = 523000", "c"
        )
    )
    reg.add(
        EvidenceItem(
            "",
            "calc",
            "calc:416161 / 391035 * 100",
            None,
            "calc",
            None,
            None,
            "416161 / 391035 * 100 = 106.4",
            "c",
        )
    )
    reg.add(EvidenceItem("", "chunk", "c2", "F", "S", 1, 1, "compared to $391,035 million", "h"))
    sources = grounding_sources(reg)
    assert ungrounded_numbers("Sales were $523,000 million", sources) == [523_000e6]
    assert ungrounded_numbers("Sales were 106.4% of the prior year", sources) == []


def test_user_year_is_not_relaxed(scripted, monkeypatch):
    queue, _ = scripted
    captured = []
    monkeypatch.setattr(tools, "_search", lambda args, config: captured.append(args) or [CHUNK])
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(answer_json())]
    from ledgerlens.retrieval.filters import SearchFilters

    run_agent("q", AgentConfig(), filters=SearchFilters(fiscal_year=2025))
    assert captured[0]["strict_year"] is True


def test_many_repairs_do_not_hit_recursion_limit(scripted):
    queue, _ = scripted
    bad = answer_json(answer="Net sales were $420 billion [E1].", value=420, scale="billion")
    queue += [_resp(tool_calls=[search_call()]), _resp("READY")] + [_resp(bad) for _ in range(5)]
    assert run_agent("q", AgentConfig(max_repairs=4)).refused


def test_quote_with_wrong_number_is_not_supported():
    text = "Total net sales were $416.2 billion in 2025."
    assert not quote_supported("net sales were $500 billion", text)
    assert quote_supported("Total net sales were $416.2 billion in 2025", text)
    assert quote_supported(
        "net sales were $416.2 billion in 2025.", "net sales were $416.2 billion in 2025"
    )


def test_dollar_or_comma_years_are_numbers():
    assert numbers_in("costs of $2,016 and 2,016 units in 2016") == [2016, 2016]


def test_refused_answers_carry_no_markers_and_bool_is_not_a_value():
    reg = EvidenceRegistry()
    reg.add(EvidenceItem("", "chunk", "c1", "F", "S", 1, 1, "t", "h"))
    out = finalize({"answer": "Not disclosed [E1].", "refused": True}, reg, [])
    assert "[" not in out.answer
    out = finalize({"answer": "x [E1]", "refused": False, "value": True, "unit": "count",
                    "citations": [{"ref": "E1", "quote": "t"}]}, reg, [])  # fmt: skip
    assert out.answer_value is None


def test_string_values_from_the_model_are_parsed():
    reg = EvidenceRegistry()
    reg.add(EvidenceItem("", "fact", "f1", "F", "S", None, None, "t", "h"))
    for raw, expected in (("21,155", 21_155e6), ("$1.5", 1.5e6), ("(3.2)", -3.2e6), ("n/a", None)):
        out = finalize(
            {"answer": "x [E1]", "refused": False, "value": raw, "scale": "million", "unit": "USD"},
            reg,
            [],
        )
        assert out.answer_value == (pytest.approx(expected) if expected is not None else None)


def test_repeated_tool_calls_are_skipped_and_end_gathering(scripted):
    queue, seen = scripted
    # The model asks for the same search three rounds in a row; dedupe stops after round 2.
    queue += [
        _resp(tool_calls=[search_call()]),
        _resp(tool_calls=[search_call()]),
        _resp(answer_json()),
    ]
    result = run_agent("q", AgentConfig(max_rounds=3))
    assert result.steps.count("search_filings") == 1
    assert [s["purpose"] for s in seen] == ["gather", "gather", "answer"]


def test_dedupe_off_reproduces_repeats(scripted):
    queue, _ = scripted
    queue += [_resp(tool_calls=[search_call()]) for _ in range(3)] + [_resp(answer_json())]
    result = run_agent("q", AgentConfig(max_rounds=3, dedupe_tool_calls=False))
    assert result.steps.count("search_filings") == 3


def test_full_number_with_a_redundant_scale_is_not_scaled_twice():
    reg = EvidenceRegistry()
    reg.add(EvidenceItem("", "fact", "f1", "F", "S", None, None, "t", "h"))
    draft = {
        "answer": "Total assets were $359,241,000,000 [E1].",
        "refused": False,
        "value": "359,241,000,000",
        "scale": "billion",
        "unit": "USD",
    }
    assert finalize(draft, reg, []).answer_value == pytest.approx(359_241e6)
    draft = {
        "answer": "Total assets were $359.2 billion [E1].",
        "refused": False,
        "value": 359.2,
        "scale": "billion",
        "unit": "USD",
    }
    assert finalize(draft, reg, []).answer_value == pytest.approx(359.2e9)


def test_derived_numbers_are_accepted_but_invented_ones_are_not():
    from ledgerlens.agent.graph import derivable

    base = [416_161e6, 391_035e6, 25.3, 50.0]
    assert derivable(6.4255, base)  # percent change 391,035 -> 416,161
    assert derivable(807_196e6, base)  # sum
    assert derivable(50.6, [25.3, 50.0])  # 25.3 / 50 * 100
    assert not derivable(777.7, base)


def test_verify_accepts_a_derived_percentage(scripted):
    queue, _ = scripted
    derived = answer_json(
        answer="Net sales grew 6.43% to $416,161 million [E1].",
        value=6.43,
        scale="one",
        unit="percent",
    )
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(derived)]
    result = run_agent("q", AgentConfig())
    assert not result.refused and result.answer_value == pytest.approx(6.43)
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(derived), _resp(derived)]
    assert run_agent("q2", AgentConfig(derive_numbers=False)).refused


def test_grouped_citation_markers_are_split():
    from ledgerlens.agent.graph import split_grouped_refs

    assert (
        split_grouped_refs("Cash was $21,846 million [E1, E4].")
        == "Cash was $21,846 million [E1][E4]."
    )
    assert split_grouped_refs("x [E2;E3] y [E5]") == "x [E2][E3] y [E5]"


def test_grouped_markers_no_longer_force_a_refusal(scripted):
    queue, _ = scripted
    grouped = answer_json(answer="Total net sales were $416,161 million in 2025 [E1, E1].")
    queue += [_resp(tool_calls=[search_call()]), _resp("READY"), _resp(grouped)]
    result = run_agent("q", AgentConfig())
    assert not result.refused and "[1]" in result.answer
