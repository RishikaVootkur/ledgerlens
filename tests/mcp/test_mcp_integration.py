"""search_filings against the real local index (SQLite, BM25, Qdrant, Ollama).

Run with: DATA_DIR=/path/to/data uv run pytest -m integration tests/mcp
"""

import pytest
from mcp.client.client import Client

from ledgerlens.mcp_server import build_server

pytestmark = pytest.mark.integration


async def test_search_filings_real_index():
    async with Client(build_server(include_ask=False)) as client:
        result = await client.call_tool(
            "search_filings", {"query": "Apple net sales", "ticker": "AAPL", "k": 3}
        )
    assert not result.is_error, result.content
    data = result.structured_content
    assert data is not None
    hits = data["results"]
    assert len(hits) == 3
    assert all(h["ticker"] == "AAPL" and h["filing_id"].startswith("AAPL-") for h in hits)
    assert all(h["chunk_id"].startswith(h["filing_id"]) for h in hits)
    assert any("net sales" in h["text"].lower() for h in hits)
