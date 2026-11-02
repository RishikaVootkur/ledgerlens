import pytest

from ledgerlens.retrieval import retriever as r
from ledgerlens.retrieval.bm25 import BM25Index
from ledgerlens.retrieval.store import ChunkStore


class FakeDense:
    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, k, filters=None):
        return [(cid, 1.0 / (i + 1)) for i, cid in enumerate(self.ranking[:k])]


@pytest.fixture
def wired(corpus, tmp_path, monkeypatch):
    store = ChunkStore(tmp_path / "chunks.sqlite")
    store.upsert(corpus)
    bm25 = BM25Index.build(corpus)
    dense_rank = [corpus[1].chunk_id, corpus[0].chunk_id]
    monkeypatch.setattr(r, "get_store", lambda: store)
    monkeypatch.setattr(r, "get_bm25", lambda s: bm25)
    monkeypatch.setattr(r, "get_dense", lambda s: FakeDense(dense_rank))
    return corpus


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_methods_return_ranked_hits(wired, method):
    hits = r.Retriever(method=method).search("iPhone net sales", k=2)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.chunk.chunk_id for h in hits)


def test_hybrid_fuses_both_lists(wired):
    hits = r.Retriever(method="hybrid").search("research and development expense", k=3)
    ids = [h.chunk.chunk_id for h in hits]
    # R&D chunk is rank 1 in both BM25 and the fake dense list, so it must win.
    assert ids[0] == wired[1].chunk_id


def test_rerank_reorders(wired, monkeypatch):
    monkeypatch.setattr(r, "rerank", lambda q, pairs, top_k: [(pairs[-1][0], 9.0)][:top_k])
    hits = r.Retriever(method="hybrid_rerank").search("iPhone", k=1)
    assert len(hits) == 1 and hits[0].score == 9.0
