"""MCP server exposing the agent's tools to any MCP client (Claude Desktop, IDEs, agents).

The tools are thin wrappers over the same functions the LangGraph agent calls
(`ledgerlens.agent.tools`, `xbrl`, `calc`), so retrieval and XBRL logic live in one place.
Every tool only reads: search and tables come from the local index, facts from SEC
companyfacts (cached on disk after the first fetch). Run with `ledgerlens mcp serve`.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from typing import Annotated, Any, Literal, NoReturn

import httpx
import typer
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from ledgerlens import __version__
from ledgerlens.agent import calc, xbrl
from ledgerlens.agent import tools as agent_tools
from ledgerlens.agent.xbrl import METRICS, Fact
from ledgerlens.llm.cache import CacheMissError
from ledgerlens.logging import get_logger
from ledgerlens.schemas import AgentAnswer, FiscalPeriod, FormType
from ledgerlens.universe import BY_TICKER, UNIVERSE

log = get_logger(__name__)

COMPANIES_URI = "ledgerlens://companies"
MAX_K = 20

INSTRUCTIONS = (
    "LedgerLens searches SEC 10-K and 10-Q filings. Use search_filings for passages (filter "
    "by ticker, fiscal_year and form whenever the question names them), get_table for the "
    "full table behind a passage, get_company_facts for exact XBRL-reported values and "
    "calculate for arithmetic. Cite passages by chunk_id, filing_id and page. Read "
    f"{COMPANIES_URI} for the covered tickers."
)

# Search, tables and arithmetic touch only the local index; facts may fetch SEC once.
LOCAL = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
REMOTE = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=True)

Duration = Literal["FY", "Q", "YTD-6M", "YTD-9M", "instant"]


class Passage(BaseModel):
    """One retrieved filing passage, with everything needed to cite it."""

    chunk_id: str
    filing_id: str
    ticker: str
    company: str
    form: FormType
    fiscal_year: int
    fiscal_period: FiscalPeriod
    doc_name: str | None = None
    section: str
    page: int | None = None
    page_end: int | None = None
    has_table: bool = False
    text: str


class SearchResult(BaseModel):
    query: str
    results: list[Passage]


class TableResult(BaseModel):
    chunk_id: str
    chunks: list[Passage] = Field(description="The passage(s) holding the table, in order")


class FactsResult(BaseModel):
    ticker: str
    metric: str
    facts: list[Fact] = Field(description="Newest period first")


class CalcResult(BaseModel):
    expression: str
    value: float


class CompanyInfo(BaseModel):
    ticker: str
    cik: str
    name: str
    sector: str


def _passages(chunks: list[dict[str, Any]]) -> list[Passage]:
    fields = Passage.model_fields
    return [Passage.model_validate({k: v for k, v in c.items() if k in fields}) for c in chunks]


def _ticker(ticker: str) -> str:
    value = ticker.strip().upper()
    if value not in BY_TICKER:
        covered = ", ".join(c.ticker for c in UNIVERSE)
        raise ToolError(f"Unknown ticker {ticker!r}. Covered tickers: {covered}")
    return value


# ------------------------------------------------------------------ failures
#
# Raw exception text can carry file paths, hostnames or provider account details, so it
# never goes to the client. Failures we expect become a ToolError with a fixed message and
# are logged here with the traceback; anything else propagates, and the SDK answers with
# its generic "Error executing tool <name>" and logs the traceback itself.

INDEX_MISSING = "The search index is not built on this server. Run `ledgerlens index build`."
SEARCH_DOWN = "The search backend is unavailable. Check that Qdrant and Ollama are running."
SEC_DOWN = "SEC companyfacts is unavailable right now. Try again later."
SEC_NOT_CONFIGURED = "These facts are not cached and SEC_USER_AGENT is not set on the server."
LLM_QUOTA = "The answer model's daily quota is exhausted. Use the other tools or retry later."
LLM_RATE_LIMITED = "The answer model is rate limited. Retry in a minute."
LLM_AUTH = "The server's LLM provider credentials were rejected."
LLM_REPLAY_MISS = "The server runs in replay mode and has no recorded answer for this question."


def _search_failure(exc: Exception) -> str | None:
    from qdrant_client.http.exceptions import ApiException

    if isinstance(exc, FileNotFoundError | sqlite3.OperationalError):
        return INDEX_MISSING
    if isinstance(exc, httpx.HTTPError | ConnectionError | ApiException):
        return SEARCH_DOWN
    return None


def _sec_failure(exc: Exception) -> str | None:
    if isinstance(exc, ValueError) and "SEC_USER_AGENT" in str(exc):
        return SEC_NOT_CONFIGURED
    if isinstance(exc, httpx.HTTPError | ConnectionError):
        return SEC_DOWN
    return None


def _llm_failure(exc: Exception) -> str | None:
    from ledgerlens.llm.provider import QuotaExhaustedError

    if isinstance(exc, QuotaExhaustedError):
        return LLM_QUOTA
    if isinstance(exc, CacheMissError):
        return LLM_REPLAY_MISS
    # LiteLLM's exception classes, matched by name so this module does not import LiteLLM.
    name = type(exc).__name__
    if name == "RateLimitError":
        return LLM_RATE_LIMITED
    if name == "AuthenticationError":
        return LLM_AUTH
    return _search_failure(exc) or _sec_failure(exc)


def _reraise_safely(
    tool: str, exc: Exception, classify: Callable[[Exception], str | None]
) -> NoReturn:
    message = classify(exc)
    if message is None:
        raise exc
    log.warning("mcp_tool_failed", tool=tool, error=type(exc).__name__, exc_info=exc)
    raise ToolError(message) from exc


# ------------------------------------------------------------------ tools


def search_filings(
    query: Annotated[str, Field(min_length=1, description="What to look for")],
    ticker: Annotated[str | None, Field(description="Company ticker, e.g. AAPL")] = None,
    fiscal_year: Annotated[
        int | None, Field(ge=1990, le=2100, description="Fiscal year, e.g. 2025")
    ] = None,
    form: Annotated[Literal["10-K", "10-Q"] | None, Field(description="Filing form")] = None,
    doc_name: Annotated[str | None, Field(description="Exact document name if known")] = None,
    k: Annotated[int, Field(ge=1, le=MAX_K, description="Number of passages")] = 6,
) -> SearchResult:
    """Search SEC 10-K/10-Q filing passages (text and tables) with hybrid BM25 + dense
    retrieval. Use focused queries that name the metric and period. If a fiscal_year filter
    finds nothing, the search is retried once without it (check each passage's year)."""
    args = {
        "query": query,
        "ticker": _ticker(ticker) if ticker else None,
        "fiscal_year": fiscal_year,
        "form": form,
        "doc_name": doc_name,
    }
    try:
        chunks = agent_tools._search(args, agent_tools.RetrievalConfig(k=k))
    except Exception as exc:
        _reraise_safely("search_filings", exc, _search_failure)
    return SearchResult(query=query, results=_passages(chunks))


def get_table(
    chunk_id: Annotated[str, Field(min_length=1, description="chunk_id from search_filings")],
) -> TableResult:
    """Return the full table(s) contained in a search result, by its chunk_id, when a
    passage shows only part of a table. A passage without a table is returned as is."""
    try:
        chunks = agent_tools._table(chunk_id.strip())
    except Exception as exc:
        _reraise_safely("get_table", exc, _search_failure)
    if not chunks:
        raise ToolError(f"Unknown chunk_id {chunk_id!r}; pass a chunk_id from search_filings.")
    return TableResult(chunk_id=chunk_id, chunks=_passages(chunks))


def get_company_facts(
    ticker: Annotated[str, Field(min_length=1, description="Company ticker, e.g. AAPL")],
    metric: Annotated[
        str,
        Field(
            min_length=1,
            description="One of: " + ", ".join(METRICS) + ", or a concept like us-gaap:Assets",
        ),
    ],
    fiscal_year: Annotated[
        int | None, Field(ge=1990, le=2100, description="Fiscal year of the reporting filing")
    ] = None,
    duration: Annotated[
        Duration | None,
        Field(description="FY, Q (single quarter), YTD-6M, YTD-9M or instant (balance sheet)"),
    ] = None,
) -> FactsResult:
    """Exact values the company reported in XBRL (SEC companyfacts). Each fact carries its
    own period dates and the filing that first reported it."""
    symbol = _ticker(ticker)
    name = metric.strip()
    if name.lower() not in METRICS and ":" not in name:
        raise ToolError(
            f"Unknown metric {metric!r}. Use one of: {', '.join(METRICS)}, "
            "or a concept like us-gaap:GrossProfit."
        )
    try:
        facts = xbrl.get_company_facts(symbol, name, fiscal_year, duration)
    except Exception as exc:
        _reraise_safely("get_company_facts", exc, _sec_failure)
    return FactsResult(ticker=symbol, metric=name, facts=facts)


def calculate(
    expression: Annotated[
        str,
        Field(min_length=1, max_length=calc.MAX_LEN, description="e.g. (416161 - 391035) / 391035"),
    ],
) -> CalcResult:
    """Evaluate arithmetic exactly. Numbers, + - * / ** ( ), % as percent, and abs round
    min max sqrt only; anything else is rejected."""
    try:
        value = calc.calculate(expression)
    except calc.CalcError as exc:
        raise ToolError(f"Invalid expression: {exc}") from exc
    return CalcResult(expression=expression, value=value)


def ask(
    question: Annotated[str, Field(min_length=1, description="A question about the filings")],
    ticker: Annotated[str | None, Field(description="Restrict to one company")] = None,
    fiscal_year: Annotated[int | None, Field(ge=1990, le=2100)] = None,
) -> AgentAnswer:
    """Run the full LedgerLens agent (search, XBRL facts, calculation, self-check) and
    return a cited answer. Slow (several LLM calls); prefer the other tools when you can
    reason over the evidence yourself."""
    from ledgerlens.agent.graph import run_agent
    from ledgerlens.retrieval.filters import SearchFilters

    filters = SearchFilters(ticker=_ticker(ticker) if ticker else None, fiscal_year=fiscal_year)
    try:
        return run_agent(question, filters=filters)
    except Exception as exc:
        _reraise_safely("ask", exc, _llm_failure)


def list_companies() -> str:
    """Companies LedgerLens covers (ticker, CIK, name, sector) as JSON."""
    companies = [CompanyInfo(**c._asdict()).model_dump() for c in UNIVERSE]
    return json.dumps(companies, indent=2)


def build_server(*, include_ask: bool = True) -> MCPServer:
    """A fresh server instance; tests build their own so state never leaks between them."""
    server: MCPServer = MCPServer(
        name="ledgerlens",
        title="LedgerLens",
        description="Cited research over SEC 10-K and 10-Q filings",
        instructions=INSTRUCTIONS,
        version=__version__,
        website_url="https://github.com/RishikaVootkur/ledgerlens",
    )
    server.add_tool(search_filings, title="Search filings", annotations=LOCAL)
    server.add_tool(get_table, title="Get table", annotations=LOCAL)
    server.add_tool(get_company_facts, title="Get XBRL company facts", annotations=REMOTE)
    server.add_tool(calculate, title="Calculate", annotations=LOCAL)
    if include_ask:
        server.add_tool(
            ask,
            title="Ask the LedgerLens agent",
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
        )
    server.resource(
        COMPANIES_URI,
        name="companies",
        title="Covered companies",
        mime_type="application/json",
    )(list_companies)
    return server


# ------------------------------------------------------------------ CLI

mcp_app = typer.Typer(help="Serve the LedgerLens tools over the Model Context Protocol.")


@mcp_app.command("serve")
def serve(
    transport: str = typer.Option("stdio", help="stdio (default) or http (streamable HTTP)"),
    host: str = typer.Option("127.0.0.1", help="HTTP bind address"),
    port: int = typer.Option(8765, help="HTTP port"),
    no_ask: bool = typer.Option(False, "--no-ask", help="Do not expose the LLM-backed ask tool"),
) -> None:
    """Run the MCP server. stdio for local clients, http serves streamable HTTP at /mcp."""
    server = build_server(include_ask=not no_ask)
    if transport == "stdio":
        server.run("stdio")
    elif transport == "http":
        typer.echo(f"LedgerLens MCP on http://{host}:{port}/mcp", err=True)
        server.run("streamable-http", host=host, port=port)
    else:
        raise typer.BadParameter("transport must be stdio or http", param_hint="--transport")
