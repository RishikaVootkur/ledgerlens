"""Agent tools: search_filings, get_table, get_company_facts, calculate.

Each tool is a plain function returning JSON-serializable data, so the same functions
back the LangGraph agent and the MCP server. Results are cached on disk keyed on the
tool name, arguments and retrieval config; in replay mode they come only from the
committed cache, which lets CI replay full agent runs without Qdrant, Ollama or SEC.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ledgerlens.agent.calc import CalcError, calculate
from ledgerlens.agent.xbrl import METRICS, get_company_facts
from ledgerlens.config import get_settings
from ledgerlens.llm.cache import CacheMissError, ResponseCache, cache_key
from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.retrieval.retriever import Method, Retriever, get_store
from ledgerlens.schemas import ChunkStrategy
from ledgerlens.universe import UNIVERSE

SNIPPET_CHARS = 2400


@dataclass(frozen=True)
class RetrievalConfig:
    strategy: ChunkStrategy = "section"
    method: Method = "hybrid"
    k: int = 6
    index_version: str = "auto"


# ------------------------------------------------------------------ tool schemas

TOOL_SPECS: dict[str, dict[str, Any]] = {
    "search_filings": {
        "description": (
            "Search SEC 10-K/10-Q filing passages (text and tables). Use focused queries "
            "that name the metric and period. Filter by company ticker, fiscal year or form "
            "whenever the question names them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for"},
                "ticker": {"type": "string", "description": "Company ticker, e.g. AAPL"},
                "fiscal_year": {"type": "integer", "description": "Fiscal year, e.g. 2025"},
                "form": {"type": "string", "enum": ["10-K", "10-Q"]},
                "doc_name": {"type": "string", "description": "Exact document name if known"},
            },
            "required": ["query"],
        },
    },
    "get_table": {
        "description": (
            "Return the full table(s) contained in a search result, by its evidence id "
            "(e.g. E3), when a snippet shows only part of a table."
        ),
        "parameters": {
            "type": "object",
            "properties": {"evidence_id": {"type": "string"}},
            "required": ["evidence_id"],
        },
    },
    "get_company_facts": {
        "description": (
            "Exact values the company reported in XBRL (SEC companyfacts). metric is one of: "
            + ", ".join(METRICS)
            + " or a concept like us-gaap:GrossProfit. duration: FY, Q (single quarter), "
            "YTD-6M, YTD-9M or instant (balance sheet)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "metric": {"type": "string"},
                "fiscal_year": {"type": "integer"},
                "duration": {
                    "type": "string",
                    "enum": ["FY", "Q", "YTD-6M", "YTD-9M", "instant"],
                },
            },
            "required": ["ticker", "metric"],
        },
    },
    "calculate": {
        "description": (
            "Evaluate arithmetic exactly, e.g. (416161 - 391035) / 391035 * 100. "
            "Numbers and + - * / ** ( ) abs round min max sqrt only."
        ),
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


def openai_tools(names: list[str]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": n, **TOOL_SPECS[n]}}
        for n in names
        if n in TOOL_SPECS
    ]


# ------------------------------------------------------------------ evidence


@dataclass
class EvidenceItem:
    ref: str  # E1, E2, ...
    kind: str  # chunk | fact | calc
    source_id: str  # chunk_id, fact_id, or calc expression
    filing_id: str | None
    section: str
    page: int | None
    page_end: int | None
    text: str
    header: str
    doc_name: str | None = None


@dataclass
class EvidenceRegistry:
    items: dict[str, EvidenceItem] = field(default_factory=dict)
    _by_source: dict[str, str] = field(default_factory=dict)

    def add(self, item: EvidenceItem) -> EvidenceItem:
        if existing := self._by_source.get(item.source_id):
            return self.items[existing]
        item.ref = f"E{len(self.items) + 1}"
        self.items[item.ref] = item
        self._by_source[item.source_id] = item.ref
        return item

    def chunk_ids(self) -> list[str]:
        return [i.source_id for i in self.items.values() if i.kind == "chunk"]


# ------------------------------------------------------------------ cached execution


def _cached_tool(
    name: str, args: dict[str, Any], config: RetrievalConfig, fn: Callable[[], Any]
) -> Any:
    s = get_settings()
    replay = s.llm_mode == "replay"
    committed = s.eval_dir / "cache"
    cache = ResponseCache(committed) if replay else ResponseCache(s.cache_dir, [committed])
    payload = {
        "tool": name,
        "args": args,
        "config": {**config.__dict__, "index_version": resolve_index_version(config)},
    }
    key = cache_key("tool", payload)
    if (hit := cache.get("tool", key)) is not None:
        return hit["result"]
    if replay:
        raise CacheMissError(f"No recorded result for tool {name} {args}")
    result = fn()
    if resolve_index_version(config) != "building":  # never cache a half-built index
        cache.put("tool", key, {**payload, "result": result})
    return result


def resolve_index_version(config: RetrievalConfig) -> str:
    """`auto` means the fingerprint the indexer wrote for this strategy, so a reindex
    invalidates cached search results. Replayed configs pin an explicit version."""
    if config.index_version != "auto":
        return config.index_version
    path = get_settings().data_dir / "index" / f"version_{config.strategy}.txt"
    return path.read_text().strip() if path.exists() else "none"


def _search(args: dict[str, Any], config: RetrievalConfig) -> list[dict[str, Any]]:
    filters = SearchFilters.model_validate(
        {k: args.get(k) for k in ("ticker", "fiscal_year", "form", "filing_id", "doc_name")}
    )
    hits = Retriever(config.strategy, config.method).search(args["query"], config.k, filters)
    if not hits and args.get("fiscal_year") and not args.get("strict_year"):
        # Fiscal years are easy to get off by one; retry once without the year.
        relaxed = filters.model_copy(update={"fiscal_year": None})
        hits = Retriever(config.strategy, config.method).search(args["query"], config.k, relaxed)
    return [h.chunk.model_dump(mode="json") for h in hits]


def _table(chunk_id: str) -> list[dict[str, Any]]:
    store = get_store()
    chunk = store.get(chunk_id)
    if chunk is None:
        return []
    if not chunk.table_ids:
        return [chunk.model_dump(mode="json")]
    siblings = store.filing_chunks(chunk.filing_id, chunk.strategy)
    wanted = set(chunk.table_ids)
    return [c.model_dump(mode="json") for c in siblings if wanted & set(c.table_ids)]


def _chunk_header(c: dict[str, Any]) -> str:
    page = f"p.{c['page']}" if c.get("page") else "p.?"
    period = f"{c['fiscal_period']} {c['fiscal_year']}"
    return f"{c['ticker']} {c['form']} {period} | {c['section'][:80]} | {page}"


def run_tool(
    name: str, args: dict[str, Any], registry: EvidenceRegistry, config: RetrievalConfig
) -> str:
    """Execute one tool call, register its evidence and return compact text for the model."""
    try:
        if name == "search_filings":
            chunks = _cached_tool(name, args, config, lambda: _search(args, config))
            return _register_chunks(chunks, registry) or "No matching passages."
        if name == "get_table":
            item = registry.items.get(str(args.get("evidence_id", "")).strip())
            if item is None or item.kind != "chunk":
                return "Unknown evidence id; pass an id like E3 from search results."
            chunk_id = item.source_id
            chunks = _cached_tool(name, {"chunk_id": chunk_id}, config, lambda: _table(chunk_id))
            return _register_chunks(chunks, registry, full=True) or "No table found."
        if name == "get_company_facts":
            ticker = str(args.get("ticker", "")).upper()
            if ticker not in {c.ticker for c in UNIVERSE}:
                return f"{ticker} is not covered. Covered: {', '.join(c.ticker for c in UNIVERSE)}"
            facts = _cached_tool(
                name,
                args,
                config,
                lambda: [
                    f.model_dump(mode="json")
                    for f in get_company_facts(
                        ticker, str(args["metric"]), args.get("fiscal_year"), args.get("duration")
                    )
                ],
            )
            return _register_facts(facts, registry) or "No reported values found."
        if name == "calculate":
            expression = str(args.get("expression", ""))
            value = calculate(expression)
            item = registry.add(
                EvidenceItem(
                    "",
                    "calc",
                    f"calc:{expression}",
                    None,
                    "calculation",
                    None,
                    None,
                    f"{expression} = {value:.6g}",
                    "calculate",
                )
            )
            return f"[{item.ref}] {expression} = {value:.10g}"
        return f"Unknown tool {name}."
    except CalcError as exc:
        return f"calculate error: {exc}"
    except CacheMissError:
        raise
    except Exception as exc:  # tool failures are reported to the model, not raised
        return f"{name} failed: {type(exc).__name__}: {str(exc)[:200]}"


def _register_chunks(
    chunks: list[dict[str, Any]], registry: EvidenceRegistry, full: bool = False
) -> str:
    lines = []
    for c in chunks:
        item = registry.add(
            EvidenceItem(
                ref="",
                kind="chunk",
                source_id=c["chunk_id"],
                filing_id=c["filing_id"],
                section=c["section"],
                page=c.get("page"),
                page_end=c.get("page_end"),
                text=c["text"],
                header=_chunk_header(c),
                doc_name=c.get("doc_name"),
            )
        )
        body = item.text if full else item.text[:SNIPPET_CHARS]
        lines.append(f"[{item.ref}] {item.header}\n{body}")
    return "\n\n".join(lines)


def _register_facts(facts: list[dict[str, Any]], registry: EvidenceRegistry) -> str:
    lines = []
    for f in facts:
        period = f"{f['start']} to {f['end']}" if f.get("start") else f"as of {f['end']}"
        text = (
            f"{f['label']} ({f['concept']}): {f['value']:,.4f} {f['unit']} for {period} "
            f"[{f['duration']}], reported in {f['form']} {f['fiscal_period']} {f['fiscal_year']}"
        ).replace(".0000 ", " ")
        item = registry.add(
            EvidenceItem(
                ref="",
                kind="fact",
                source_id=f["fact_id"],
                filing_id=f.get("filing_id"),
                section="XBRL companyfacts",
                page=None,
                page_end=None,
                text=text,
                header=f"{f['ticker']} XBRL",
            )
        )
        lines.append(f"[{item.ref}] {text}")
    return "\n".join(lines)


def parse_arguments(raw: str) -> dict[str, Any]:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}
