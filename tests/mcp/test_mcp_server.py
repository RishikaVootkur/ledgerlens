"""Drive the MCP server through a real MCP client session, with no network.

`Client(server)` connects in process; `Client(InMemoryTransport(server))` sends every
message as JSON-RPC over in-memory streams, so both the direct and the wire path are
covered. Retrieval and SEC companyfacts are replaced with fakes.
"""

import json
import os
import sys
from collections.abc import Callable
from typing import Any, ClassVar

import httpx
import pytest
from mcp.client._memory import InMemoryTransport
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types.version import LATEST_HANDSHAKE_VERSION, is_version_at_least

from ledgerlens import mcp_server
from ledgerlens.agent import tools as agent_tools
from ledgerlens.agent import xbrl
from ledgerlens.config import get_settings
from ledgerlens.llm.cache import CacheMissError
from ledgerlens.llm.provider import QuotaExhaustedError
from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.retrieval.retriever import Hit
from ledgerlens.schemas import Chunk
from ledgerlens.universe import UNIVERSE

MIN_PROTOCOL = "2026-07-28"
TOOLS = {"search_filings", "get_table", "get_company_facts", "calculate", "ask"}

CF = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "label": "Revenue",
                "units": {"USD": [
                    {"start": "2024-09-29", "end": "2025-09-27", "val": 416161e6, "accn": "a2",
                     "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2025-10-31"},
                    {"start": "2023-10-01", "end": "2024-09-28", "val": 391035e6, "accn": "a1",
                     "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-01"},
                    {"start": "2025-03-30", "end": "2025-06-28", "val": 94e9, "accn": "a4",
                     "fy": 2025, "fp": "Q3", "form": "10-Q", "filed": "2025-08-01"},
                ]},
            },
        }
    }
}  # fmt: skip


def chunk(seq: int, text: str, table_ids: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"AAPL-10K-2025FY:section:{seq:05d}",
        filing_id="AAPL-10K-2025FY",
        ticker="AAPL",
        company="Apple Inc.",
        form="10-K",
        fiscal_year=2025,
        fiscal_period="FY",
        strategy="section",
        section="Item 8. Financial Statements",
        page=28,
        page_end=28,
        text=text,
        has_table=bool(table_ids),
        table_ids=table_ids or [],
    )


CHUNKS = [
    chunk(1, "Total net sales were $416,161 million in 2025.", ["t1"]),
    chunk(2, "| Net sales | 416,161 | 391,035 |", ["t1"]),
    chunk(3, "Research and development expense was $34.5 billion."),
]


class FakeRetriever:
    calls: ClassVar[list[tuple[str, int, SearchFilters | None]]] = []

    def __init__(self, strategy: str = "section", method: str = "hybrid") -> None:
        self.strategy, self.method = strategy, method

    def search(self, query: str, k: int = 8, filters: SearchFilters | None = None) -> list[Hit]:
        FakeRetriever.calls.append((query, k, filters))
        return [Hit(chunk=c, score=1.0 / i, rank=i) for i, c in enumerate(CHUNKS[:k], start=1)]


class FakeStore:
    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in CHUNKS if c.chunk_id == chunk_id), None)

    def filing_chunks(self, filing_id: str, strategy: str) -> list[Chunk]:
        return [c for c in CHUNKS if c.filing_id == filing_id]


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    get_settings.cache_clear()
    xbrl._companyfacts.cache_clear()
    FakeRetriever.calls = []
    monkeypatch.setattr(agent_tools, "Retriever", FakeRetriever)
    monkeypatch.setattr(agent_tools, "get_store", FakeStore)
    monkeypatch.setattr(xbrl, "_fetch_companyfacts", lambda cik: CF)
    yield
    get_settings.cache_clear()
    xbrl._companyfacts.cache_clear()


TARGETS: dict[str, Callable[[], Any]] = {
    "direct": mcp_server.build_server,
    "jsonrpc": lambda: InMemoryTransport(mcp_server.build_server()),
}


@pytest.fixture(params=sorted(TARGETS))
def connect(request) -> Callable[[], Client]:
    """A client factory. Tests open the session themselves: anyio cancel scopes must be
    entered and exited in the same task, which an async pytest fixture does not guarantee."""
    return lambda: Client(TARGETS[request.param]())


async def test_negotiates_current_protocol(connect):
    async with connect() as client:
        assert is_version_at_least(client.protocol_version, MIN_PROTOCOL)
        assert client.server_info is not None and client.server_info.name == "ledgerlens"


async def test_lists_tools_with_schemas_and_read_only_annotations(connect):
    async with connect() as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == TOOLS
        for tool in tools.values():
            assert tool.annotations is not None and tool.annotations.read_only_hint is True
            assert tool.output_schema is not None, tool.name
        search = tools["search_filings"].input_schema
        assert search["required"] == ["query"]
        assert set(search["properties"]) == {
            "query",
            "ticker",
            "fiscal_year",
            "form",
            "doc_name",
            "k",
        }
        assert tools["search_filings"].annotations.open_world_hint is False
        assert tools["get_table"].input_schema["required"] == ["chunk_id"]
        facts = tools["get_company_facts"].input_schema
        assert facts["required"] == ["ticker", "metric"]
        assert set(facts["properties"]) == {"ticker", "metric", "fiscal_year", "duration", "limit"}
        assert tools["calculate"].input_schema["required"] == ["expression"]


async def test_calculate(connect):
    async with connect() as client:
        result = await client.call_tool(
            "calculate", {"expression": "(416,161 - 391,035) / 391,035"}
        )
        assert not result.is_error
        assert result.structured_content is not None
        assert result.structured_content["value"] == pytest.approx(0.0642551, rel=1e-5)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "(1).__class__.__bases__",
        "9 ** 9 ** 9",
    ],
)
async def test_calculate_rejects_injection(connect, expression):
    async with connect() as client:
        result = await client.call_tool("calculate", {"expression": expression})
        assert result.is_error
        assert result.content[0].text.startswith(
            "Error executing tool calculate: Invalid expression"
        )


async def test_calculate_validates_input_length(connect):
    async with connect() as client:
        result = await client.call_tool("calculate", {"expression": "1+" * 200 + "1"})
        assert result.is_error


async def test_get_company_facts_from_canned_payload(connect):
    async with connect() as client:
        args = {"ticker": "aapl", "metric": "revenue", "duration": "FY"}
        result = await client.call_tool("get_company_facts", args)
        assert not result.is_error
        data = result.structured_content
        assert data is not None and data["ticker"] == "AAPL"
        assert [(f["value"], f["fiscal_year"]) for f in data["facts"]] == [
            (416161e6, 2025),
            (391035e6, 2024),
        ]
        assert data["facts"][0]["filing_id"] == "AAPL-10K-2025FY"


async def test_get_company_facts_errors(connect):
    async with connect() as client:
        unknown = await client.call_tool(
            "get_company_facts", {"ticker": "ZZZZ", "metric": "revenue"}
        )
        assert unknown.is_error and "Unknown ticker 'ZZZZ'" in unknown.content[0].text
        bad_metric = await client.call_tool(
            "get_company_facts", {"ticker": "AAPL", "metric": "vibes"}
        )
        assert bad_metric.is_error and "Unknown metric" in bad_metric.content[0].text
        bad_duration = await client.call_tool(
            "get_company_facts", {"ticker": "AAPL", "metric": "revenue", "duration": "decade"}
        )
        assert bad_duration.is_error


async def test_search_filings_with_fake_retriever(connect):
    async with connect() as client:
        args = {"query": "Apple net sales", "ticker": "aapl", "fiscal_year": 2025, "form": "10-K"}
        result = await client.call_tool("search_filings", {**args, "k": 2})
        assert not result.is_error
        data = result.structured_content
        assert data is not None and data["query"] == "Apple net sales"
        first = data["results"][0]
        assert first == {
            "chunk_id": "AAPL-10K-2025FY:section:00001",
            "filing_id": "AAPL-10K-2025FY",
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "form": "10-K",
            "fiscal_year": 2025,
            "fiscal_period": "FY",
            "doc_name": None,
            "section": "Item 8. Financial Statements",
            "page": 28,
            "page_end": 28,
            "has_table": True,
            "text": "Total net sales were $416,161 million in 2025.",
        }
        assert len(data["results"]) == 2
        query, k, filters = FakeRetriever.calls[0]
        assert (query, k) == ("Apple net sales", 2)
        assert filters == SearchFilters(ticker="AAPL", fiscal_year=2025, form="10-K")


async def test_search_filings_validates_arguments(connect):
    async with connect() as client:
        unknown = await client.call_tool("search_filings", {"query": "x", "ticker": "TSLA"})
        assert unknown.is_error and "Covered tickers" in unknown.content[0].text
        for bad in ({"query": "x", "k": 0}, {"query": "x", "form": "8-K"}, {"query": ""}):
            assert (await client.call_tool("search_filings", bad)).is_error, bad
        assert FakeRetriever.calls == []


SECRET = "/home/ops/.ledgerlens/index sk-live-1234 10.0.0.5"


def failing(exc: Exception) -> Callable[..., Any]:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return boom


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (ConnectionError(SECRET), mcp_server.SEARCH_DOWN),
        (httpx.ConnectError(SECRET), mcp_server.SEARCH_DOWN),
        (FileNotFoundError(SECRET), mcp_server.INDEX_MISSING),
    ],
)
async def test_expected_search_failures_get_safe_messages(connect, monkeypatch, exc, message):
    monkeypatch.setattr(agent_tools, "_search", failing(exc))
    async with connect() as client:
        result = await client.call_tool("search_filings", {"query": "net sales"})
    assert result.is_error
    assert result.content[0].text == f"Error executing tool search_filings: {message}"


async def test_unexpected_failures_reveal_nothing(connect, monkeypatch):
    monkeypatch.setattr(agent_tools, "_search", failing(RuntimeError(SECRET)))
    monkeypatch.setattr(agent_tools, "_table", failing(KeyError(SECRET)))
    monkeypatch.setattr(xbrl, "_fetch_companyfacts", failing(TypeError(SECRET)))
    calls = {
        "search_filings": {"query": "net sales"},
        "get_table": {"chunk_id": "x"},
        "get_company_facts": {"ticker": "AAPL", "metric": "revenue"},
    }
    async with connect() as client:
        for name, args in calls.items():
            result = await client.call_tool(name, args)
            assert result.is_error
            assert result.content[0].text == f"Error executing tool {name}", name


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (httpx.ConnectError(SECRET), mcp_server.SEC_DOWN),
        (ValueError(f"Set SEC_USER_AGENT in .env {SECRET}"), mcp_server.SEC_NOT_CONFIGURED),
    ],
)
async def test_expected_sec_failures_get_safe_messages(connect, monkeypatch, exc, message):
    monkeypatch.setattr(xbrl, "_fetch_companyfacts", failing(exc))
    async with connect() as client:
        result = await client.call_tool(
            "get_company_facts", {"ticker": "AAPL", "metric": "revenue"}
        )
    assert result.content[0].text == f"Error executing tool get_company_facts: {message}"


class AuthenticationError(Exception):
    """Stands in for LiteLLM's class of the same name."""


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (QuotaExhaustedError(SECRET), mcp_server.LLM_QUOTA),
        (AuthenticationError(SECRET), mcp_server.LLM_AUTH),
        (CacheMissError(SECRET), mcp_server.LLM_REPLAY_MISS),
        (httpx.ConnectError(SECRET), mcp_server.SEARCH_DOWN),
    ],
)
async def test_expected_ask_failures_get_safe_messages(connect, monkeypatch, exc, message):
    from ledgerlens.agent import graph

    monkeypatch.setattr(graph, "run_agent", failing(exc))
    async with connect() as client:
        result = await client.call_tool("ask", {"question": "Apple FY2025 net sales?"})
    assert result.content[0].text == f"Error executing tool ask: {message}"


async def test_get_table_returns_every_chunk_of_the_table(connect):
    async with connect() as client:
        result = await client.call_tool("get_table", {"chunk_id": CHUNKS[0].chunk_id})
        assert not result.is_error
        data = result.structured_content
        assert data is not None
        assert [c["chunk_id"] for c in data["chunks"]] == [CHUNKS[0].chunk_id, CHUNKS[1].chunk_id]
        padded = await client.call_tool("get_table", {"chunk_id": f"  {CHUNKS[2].chunk_id}\n"})
        assert padded.structured_content is not None
        assert padded.structured_content["chunk_id"] == CHUNKS[2].chunk_id
        missing = await client.call_tool("get_table", {"chunk_id": "nope"})
        assert missing.is_error and "Unknown chunk_id" in missing.content[0].text


async def test_ask_wraps_the_agent(connect, monkeypatch):
    async with connect() as client:
        from ledgerlens.agent import graph
        from ledgerlens.schemas import AgentAnswer

        seen: dict[str, Any] = {}

        def fake_run_agent(question: str, filters: SearchFilters | None = None) -> AgentAnswer:
            seen.update(question=question, filters=filters)
            return AgentAnswer(answer="Net sales were $416.2 billion [1].")

        monkeypatch.setattr(graph, "run_agent", fake_run_agent)
        result = await client.call_tool(
            "ask", {"question": "Apple FY2025 net sales?", "ticker": "aapl"}
        )
        assert not result.is_error
        assert result.structured_content is not None
        assert result.structured_content["answer"].startswith("Net sales")
        assert seen["filters"] == SearchFilters(ticker="AAPL")


async def test_companies_resource(connect):
    async with connect() as client:
        resources = (await client.list_resources()).resources
        assert [str(r.uri) for r in resources] == [mcp_server.COMPANIES_URI]
        contents = (await client.read_resource(mcp_server.COMPANIES_URI)).contents
        companies = json.loads(contents[0].text)
        assert [c["ticker"] for c in companies] == [c.ticker for c in UNIVERSE]
        assert companies[0] == {
            "ticker": "AAPL",
            "cik": "0000320193",
            "name": "Apple Inc.",
            "sector": "Information Technology",
        }


async def test_no_ask_flag_hides_the_agent_tool():
    async with Client(mcp_server.build_server(include_ask=False)) as c:
        assert {t.name for t in (await c.list_tools()).tools} == TOOLS - {"ask"}


async def test_stdio_cli_serves_over_a_subprocess(tmp_path):
    """`ledgerlens mcp serve` over real stdio: nothing but JSON-RPC may reach stdout."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from ledgerlens.cli import app; app()", "mcp", "serve"],
        env={**os.environ, "DATA_DIR": str(tmp_path)},
    )
    async with Client(params) as c:
        assert is_version_at_least(c.protocol_version, MIN_PROTOCOL)
        assert {t.name for t in (await c.list_tools()).tools} == TOOLS
        result = await c.call_tool("calculate", {"expression": "2 ** 10"})
        assert result.structured_content == {"expression": "2 ** 10", "value": 1024.0}


async def test_legacy_handshake_clients_still_connect():
    """Clients that predate 2026-07-28 negotiate through `initialize` and still work."""
    server = mcp_server.build_server(include_ask=False)
    async with Client(InMemoryTransport(server), mode="legacy") as c:
        assert c.protocol_version == LATEST_HANDSHAKE_VERSION
        result = await c.call_tool("calculate", {"expression": "1 + 1"})
        assert result.structured_content == {"expression": "1 + 1", "value": 2.0}


async def test_get_company_facts_limit(connect):
    async with connect() as client:
        one = await client.call_tool(
            "get_company_facts", {"ticker": "AAPL", "metric": "revenue", "limit": 1}
        )
        assert one.structured_content is not None
        assert [f["value"] for f in one.structured_content["facts"]] == [416161e6]
        too_many = {"ticker": "AAPL", "metric": "revenue", "limit": mcp_server.MAX_FACTS + 1}
        assert (await client.call_tool("get_company_facts", too_many)).is_error
