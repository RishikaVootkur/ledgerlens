"""Real Qdrant + Ollama + cross-encoder. Run with: pytest -m integration"""

import pytest

from ledgerlens.retrieval.dense import DenseIndex
from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.retrieval.rerank import rerank

pytestmark = pytest.mark.integration


def test_dense_search_and_rerank_end_to_end(corpus):
    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://localhost:6333")
    index = DenseIndex(client, "fixed")
    index.collection = "ledgerlens_test"
    if client.collection_exists(index.collection):
        client.delete_collection(index.collection)
    index.ensure_collection()
    index.add(corpus)
    hits = index.search("How much did Apple spend on R&D?", k=2)
    assert hits[0][0] == corpus[1].chunk_id
    ko = index.search("volume growth", k=5, filters=SearchFilters(ticker="KO"))
    assert [cid for cid, _ in ko] == [corpus[3].chunk_id]
    ranked = rerank(
        "research and development expense", [(c.chunk_id, c.text) for c in corpus], top_k=1
    )
    assert ranked[0][0] == corpus[1].chunk_id
    client.delete_collection(index.collection)
