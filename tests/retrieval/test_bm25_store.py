from ledgerlens.retrieval.bm25 import BM25Index, tokenize
from ledgerlens.retrieval.dense import point_id, to_qdrant_filter
from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.retrieval.store import ChunkStore


def test_tokenize_keeps_numbers_and_drops_stopwords():
    assert tokenize("What was the $1,234.5 million in FY2025?") == ["1,234.5", "million", "fy2025"]


def test_bm25_ranks_and_filters(corpus, tmp_path):
    index = BM25Index.build(corpus)
    top = index.search("iPhone net sales", k=3)
    assert top[0][0].endswith("00001") or top[0][0].endswith("00005")
    only_2024 = index.search("iPhone net sales", k=3, filters=SearchFilters(fiscal_year=2024))
    assert [cid for cid, _ in only_2024] == ["AAPL-10K-2024FY:section:00005"]
    jpm = index.search("revenue", k=5, filters=SearchFilters(ticker="jpm"))
    assert all("JPM" in cid for cid, _ in jpm)
    assert index.search("zzzz unknownword", k=3) == []
    index.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25")
    assert loaded.search("iPhone net sales", k=3) == top


def test_store_round_trip_and_pending_embeddings(corpus, tmp_path):
    store = ChunkStore(tmp_path / "chunks.sqlite")
    assert store.upsert(corpus) == 5
    assert store.count("section") == 5
    assert store.get(corpus[0].chunk_id) == corpus[0]
    assert store.get("missing") is None
    assert set(store.get_many([c.chunk_id for c in corpus[:2]])) == {c.chunk_id for c in corpus[:2]}
    assert len(store.pending_embeddings("section")) == 5
    store.mark_embedded([corpus[0].chunk_id])
    assert len(store.pending_embeddings("section")) == 4
    # Changing a chunk's text makes it pending again.
    store.upsert([corpus[0].model_copy(update={"text": "changed"})])
    assert len(store.pending_embeddings("section")) == 5
    assert [c.chunk_id for c in store.filing_chunks("KO-10K-2025FY", "section")] == [
        corpus[3].chunk_id
    ]
    assert {f["ticker"] for f in store.filings()} == {"AAPL", "JPM", "KO"}


def test_qdrant_filter_and_point_ids():
    assert to_qdrant_filter(SearchFilters()) is None
    f = to_qdrant_filter(SearchFilters(ticker="aapl", fiscal_year=2025))
    assert f is not None and len(f.must) == 2
    assert f.must[0].match.value == "AAPL"
    assert point_id("x") == point_id("x") != point_id("y")
