from datetime import date

import pytest

from ledgerlens.agent import tools, xbrl
from ledgerlens.agent.tools import (
    EvidenceItem,
    EvidenceRegistry,
    RetrievalConfig,
    openai_tools,
    run_tool,
)
from ledgerlens.config import get_settings
from ledgerlens.llm.cache import CacheMissError

CF = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "label": "Revenues",
                "units": {"USD": [
                    {"start": "2016-09-25", "end": "2017-09-30", "val": 229e9, "accn": "a1",
                     "fy": 2017, "fp": "FY", "form": "10-K", "filed": "2017-11-03"},
                ]},
            },
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "label": "Revenue",
                "units": {"USD": [
                    {"start": "2024-09-29", "end": "2025-09-27", "val": 416e9, "accn": "a2",
                     "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2025-10-31"},
                    # Same period re-reported as a comparative a year later: keep the first.
                    {"start": "2024-09-29", "end": "2025-09-27", "val": 416e9, "accn": "a3",
                     "fy": 2026, "fp": "FY", "form": "10-K", "filed": "2026-10-30"},
                    {"start": "2025-03-30", "end": "2025-06-28", "val": 94e9, "accn": "a4",
                     "fy": 2025, "fp": "Q3", "form": "10-Q", "filed": "2025-08-01"},
                    {"start": "2024-09-29", "end": "2025-06-28", "val": 313e9, "accn": "a4",
                     "fy": 2025, "fp": "Q3", "form": "10-Q", "filed": "2025-08-01"},
                    {"start": "2024-09-29", "end": "2025-09-27", "val": 1.0, "accn": "x",
                     "fy": 2025, "fp": "FY", "form": "8-K", "filed": "2025-10-30"},
                ]},
            },
            "Assets": {"label": "Assets", "units": {"USD": [
                {"end": "2025-09-27", "val": 359e9, "accn": "a2", "fy": 2025, "fp": "FY",
                 "form": "10-K", "filed": "2025-10-31"}]}},
        }
    }
}  # fmt: skip


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_MODE", "live")
    get_settings.cache_clear()
    xbrl._companyfacts.cache_clear()
    monkeypatch.setattr(xbrl, "_fetch_companyfacts", lambda cik: CF)
    yield tmp_path
    get_settings.cache_clear()
    xbrl._companyfacts.cache_clear()


def test_facts_merge_concepts_dedupe_and_label_durations(isolated):
    facts = xbrl.get_company_facts("aapl", "revenue")
    assert [(f.value, f.duration) for f in facts] == [
        (416e9, "FY"),
        (94e9, "Q"),
        (313e9, "YTD-9M"),
        (229e9, "FY"),
    ]
    assert facts[0].filing_id == "AAPL-10K-2025FY" and facts[0].filed == date(2025, 10, 31)
    assert [f.value for f in xbrl.get_company_facts("AAPL", "revenue", duration="Q")] == [94e9]
    assert [f.value for f in xbrl.get_company_facts("AAPL", "revenue", fiscal_year=2017)] == [229e9]
    assert xbrl.get_company_facts("AAPL", "total_assets")[0].duration == "instant"
    assert xbrl.get_company_facts("AAPL", "us-gaap:Assets")[0].value == 359e9
    assert xbrl.get_company_facts("AAPL", "gross_profit") == []
    with pytest.raises(KeyError):
        xbrl.get_company_facts("ZZZZ", "revenue")


def test_run_tool_registers_facts_and_calc(isolated):
    reg = EvidenceRegistry()
    cfg = RetrievalConfig()
    out = run_tool(
        "get_company_facts", {"ticker": "AAPL", "metric": "revenue", "duration": "FY"}, reg, cfg
    )
    assert out.startswith("[E1]") and "416,000,000,000 USD" in out
    calc = run_tool("calculate", {"expression": "(416 - 391) / 391 * 100"}, reg, cfg)
    assert calc.startswith("[E3]") and "6.39386" in calc
    assert run_tool("calculate", {"expression": "import os"}, reg, cfg).startswith(
        "calculate error"
    )
    assert "not covered" in run_tool(
        "get_company_facts", {"ticker": "TSLA", "metric": "revenue"}, reg, cfg
    )
    assert run_tool("nope", {}, reg, cfg) == "Unknown tool nope."


def test_search_results_are_cached_and_replayable(isolated, monkeypatch):
    chunk = {"chunk_id": "KO-10K-2025FY:section:00001", "filing_id": "KO-10K-2025FY", "ticker": "KO",
             "form": "10-K", "fiscal_year": 2025, "fiscal_period": "FY", "section": "Item 7",
             "page": 3, "page_end": 3, "text": "Unit case volume grew 2%."}  # fmt: skip
    calls = []
    monkeypatch.setattr(tools, "_search", lambda args, config: calls.append(1) or [chunk])
    args = {"query": "unit case volume", "ticker": "KO"}
    first = run_tool("search_filings", args, EvidenceRegistry(), RetrievalConfig())
    second = run_tool("search_filings", args, EvidenceRegistry(), RetrievalConfig())
    assert first == second and len(calls) == 1
    # Replay mode reads only the committed eval/cache: promote the entry and replay.
    local = next((isolated / "cache" / "tool").rglob("*.json"))
    dest = isolated / "eval" / "cache" / "tool" / local.parent.name / local.name
    dest.parent.mkdir(parents=True)
    dest.write_text(local.read_text())
    monkeypatch.setenv("LLM_MODE", "replay")
    get_settings.cache_clear()
    assert run_tool("search_filings", args, EvidenceRegistry(), RetrievalConfig()) == first
    with pytest.raises(CacheMissError):
        run_tool("search_filings", {"query": "other"}, EvidenceRegistry(), RetrievalConfig())


def test_get_table_needs_known_chunk(isolated, monkeypatch):
    reg = EvidenceRegistry()
    assert "Unknown evidence id" in run_tool(
        "get_table", {"evidence_id": "E9"}, reg, RetrievalConfig()
    )


def test_tool_specs_are_valid_openai_functions():
    specs = openai_tools(["search_filings", "get_table", "get_company_facts", "calculate", "x"])
    assert [s["function"]["name"] for s in specs] == [
        "search_filings",
        "get_table",
        "get_company_facts",
        "calculate",
    ]
    assert all(s["function"]["parameters"]["type"] == "object" for s in specs)


def test_per_share_values_use_latest_restated_report(isolated, monkeypatch):
    cf = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "label": "EPS",
                    "units": {
                        "USD/shares": [
                            {
                                "start": "2018-09-30",
                                "end": "2019-09-28",
                                "val": 11.89,
                                "accn": "a",
                                "fy": 2019,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2019-10-31",
                            },
                            {
                                "start": "2018-09-30",
                                "end": "2019-09-28",
                                "val": 2.97,
                                "accn": "b",
                                "fy": 2021,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2021-10-29",
                            },
                        ]
                    },
                }
            }
        }
    }
    xbrl._companyfacts.cache_clear()
    monkeypatch.setattr(xbrl, "_fetch_companyfacts", lambda cik: cf)
    facts = xbrl.get_company_facts("AAPL", "eps_diluted")
    assert [f.value for f in facts] == [2.97]
    assert facts[0].fact_id == "xbrl:AAPL:EarningsPerShareDiluted:2018-09-30_2019-09-28:USD/shares"


def test_index_version_auto_reads_fingerprint(isolated, monkeypatch, tmp_path):
    from ledgerlens.agent.tools import resolve_index_version

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    assert resolve_index_version(RetrievalConfig()) == "none"
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "version_section.txt").write_text("abc123\n")
    assert resolve_index_version(RetrievalConfig()) == "abc123"
    assert resolve_index_version(RetrievalConfig(index_version="pinned")) == "pinned"


def test_registry_refs_are_never_reused_after_drop():
    reg = EvidenceRegistry()
    for i in range(3):
        reg.add(EvidenceItem("", "chunk", f"s{i}", "F", "S", 1, 1, "t", "h"))
    reg.drop("E2")
    item = reg.add(EvidenceItem("", "chunk", "s9", "F", "S", 1, 1, "t", "h"))
    assert item.ref == "E4" and set(reg.items) == {"E1", "E3", "E4"}
