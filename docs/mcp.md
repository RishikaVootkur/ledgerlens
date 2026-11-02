# MCP server

LedgerLens exposes the agent's tools over the [Model Context Protocol](https://modelcontextprotocol.io), so Claude Desktop, IDEs and other agents can search the filings, pull tables, read XBRL facts and do exact arithmetic with the same code the LangGraph agent uses. The server lives in `src/ledgerlens/mcp_server.py` and is built on the official Python SDK (`mcp` 2.3, `MCPServer`). It speaks protocol revision 2026-07-28 and still accepts clients that use the older `initialize` handshake (2025-11-25).

## Running it

```bash
uv run ledgerlens mcp serve                                  # stdio (default)
uv run ledgerlens mcp serve --transport http --port 8765     # streamable HTTP at http://127.0.0.1:8765/mcp
uv run ledgerlens mcp serve --no-ask                         # hide the LLM-backed ask tool
```

Search needs the local index (`data/index`), Qdrant on `localhost:6333` and Ollama on `localhost:11434` for query embeddings; point `DATA_DIR` at another data folder if needed. `get_company_facts` reads SEC companyfacts from `data/raw/sec_api/` and fetches it once from SEC on a miss (needs `SEC_USER_AGENT`). `ask` needs the answer model's API key. HTTP binds to `127.0.0.1` by default; pass `--host` to change it.

## Client configuration (stdio)

Claude Desktop (`claude_desktop_config.json`) or any client that launches stdio servers:

```json
{
  "mcpServers": {
    "ledgerlens": {
      "command": "uv",
      "args": ["--directory", "/path/to/ledgerlens", "run", "ledgerlens", "mcp", "serve"],
      "env": { "DATA_DIR": "/path/to/ledgerlens/data" }
    }
  }
}
```

GUI apps do not inherit your shell `PATH`, so use the absolute path to `uv` (`which uv`) if the client cannot find it. `env` is optional when the data lives in the repo's `data/` folder. Logs go to stderr; stdout carries only JSON-RPC.

For the HTTP transport, point the client at `http://127.0.0.1:8765/mcp`.

## Tools

All tools are annotated `readOnlyHint: true` and return structured content with an output schema. Bad input (unknown ticker, unknown metric, invalid form, `k` out of range, a non-arithmetic expression) comes back as a tool error (`isError: true`) with a message the model can act on.

| Tool | Arguments | Returns |
| --- | --- | --- |
| `search_filings` | `query`, `ticker?`, `fiscal_year?`, `form?` (10-K or 10-Q), `doc_name?`, `k?` (1 to 20, default 6) | `results`: passages with `chunk_id`, `filing_id`, `ticker`, `company`, `form`, `fiscal_year`, `fiscal_period`, `section`, `page`, `page_end`, `has_table`, `text` |
| `get_table` | `chunk_id` | every passage holding the table(s) in that chunk |
| `get_company_facts` | `ticker`, `metric` (friendly name such as `revenue` or a concept such as `us-gaap:Assets`), `fiscal_year?`, `duration?` (FY, Q, YTD-6M, YTD-9M, instant) | `facts`: XBRL values with period dates, unit and the filing that first reported them |
| `calculate` | `expression` | `value`; only numbers, `+ - * / ** ( )`, `%` and `abs round min max sqrt` are accepted |
| `ask` | `question`, `ticker?`, `fiscal_year?` | the full agent's cited `AgentAnswer` (several LLM calls) |

Search uses the default retrieval setup (section chunks, hybrid BM25 plus dense with reciprocal rank fusion), the same as the agent. As in the agent, a `fiscal_year` filter that finds nothing is retried once without the year, so check each passage's `fiscal_year`.

Resource `ledgerlens://companies` lists the covered companies (ticker, CIK, name, sector) as JSON.

## Transcript

A real call over stdio against the local AAPL/JPM/KO index, made with the SDK client launching `uv --directory <repo> run ledgerlens mcp serve`. Passage text is trimmed to 240 characters here.

```text
protocol_version: 2026-07-28
server: ledgerlens 0.1.0
tools: ['search_filings', 'get_table', 'get_company_facts', 'calculate', 'ask']
call search_filings {"query": "Apple net sales", "ticker": "AAPL", "k": 3}
is_error: False
```

```json
{
  "query": "Apple net sales",
  "results": [
    {
      "chunk_id": "AAPL-10K-2025FY:section:00080",
      "filing_id": "AAPL-10K-2025FY",
      "ticker": "AAPL",
      "company": "Apple Inc.",
      "form": "10-K",
      "fiscal_year": 2025,
      "fiscal_period": "FY",
      "doc_name": null,
      "section": "Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations > Products and Services Performance",
      "page": 25,
      "page_end": 25,
      "has_table": false,
      "text": "[AAPL 10-K FY2025 | Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations > Products and Services Performance | p.25]\n(1)Services net sales include amortization of the deferred value of services bundl ..."
    },
    {
      "chunk_id": "AAPL-10Q-2026Q2:section:00062",
      "filing_id": "AAPL-10Q-2026Q2",
      "ticker": "AAPL",
      "company": "Apple Inc.",
      "form": "10-Q",
      "fiscal_year": 2026,
      "fiscal_period": "Q2",
      "doc_name": null,
      "section": "Part I > Item 2. Management’s Discussion and Analysis of Financial Condition and Results of Operations > Products and Services Performance",
      "page": 17,
      "page_end": 17,
      "has_table": false,
      "text": "[AAPL 10-Q Q2 FY2026 | Part I > Item 2. Management’s Discussion and Analysis of Financial Condition and Results of Operations > Products and Services Performance | p.17]\niPhone\n\niPhone net sales increased during the second quarter and first ..."
    },
    {
      "chunk_id": "AAPL-10Q-2026Q3:section:00063",
      "filing_id": "AAPL-10Q-2026Q3",
      "ticker": "AAPL",
      "company": "Apple Inc.",
      "form": "10-Q",
      "fiscal_year": 2026,
      "fiscal_period": "Q3",
      "doc_name": null,
      "section": "Part I > Item 2. Management’s Discussion and Analysis of Financial Condition and Results of Operations > Products and Services Performance",
      "page": 17,
      "page_end": 17,
      "has_table": false,
      "text": "[AAPL 10-Q Q3 FY2026 | Part I > Item 2. Management’s Discussion and Analysis of Financial Condition and Results of Operations > Products and Services Performance | p.17]\niPhone\n\niPhone net sales increased during the third quarter and first  ..."
    }
  ]
}
```

The same call over `--transport http` negotiates 2026-07-28 and returns the same passages.

## Tests

`tests/mcp/test_mcp_server.py` drives the server through the SDK `Client`, both in process and as JSON-RPC over in-memory streams, plus one real stdio subprocess of `ledgerlens mcp serve` and one legacy-handshake client. Retrieval, the chunk store and companyfacts are faked, so the suite needs no network. `tests/mcp/test_mcp_integration.py` runs the search above against the real index:

```bash
uv run pytest -m integration tests/mcp
```
