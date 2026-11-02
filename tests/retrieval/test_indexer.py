import threading

import pytest

from ledgerlens.config import get_settings
from ledgerlens.retrieval import indexer, retriever
from ledgerlens.retrieval.bm25 import BM25Index
from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.retrieval.store import ChunkStore


class FakeDense:
    def __init__(self):
        self.points: set[str] = set()
        self.created = False

    def ensure_collection(self):
        was_new = not self.created
        self.created = True
        return was_new

    def count(self):
        return len(self.points)

    def add(self, chunks):
        self.points |= {c.chunk_id for c in chunks}

    def delete(self, ids):
        self.points -= set(ids)


@pytest.fixture
def env(tmp_path, monkeypatch, corpus):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    retriever.get_store.cache_clear()
    retriever.get_bm25.cache_clear()
    dense = FakeDense()
    monkeypatch.setattr(indexer, "get_dense", lambda s: dense)
    folder = tmp_path / "chunks" / "section"
    folder.mkdir(parents=True)

    def write(chunks):
        by_filing = {}
        for c in chunks:
            by_filing.setdefault(c.filing_id, []).append(c)
        for fid, cs in by_filing.items():
            (folder / f"{fid}.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in cs))

    write(corpus)
    yield dense, write, folder
    get_settings.cache_clear()
    retriever.get_store.cache_clear()
    retriever.get_bm25.cache_clear()


def test_build_is_idempotent_and_resumable(env, corpus):
    dense, _, _ = env
    first = indexer.build_index("section")
    assert first["embedded"] == 5 and first["bm25_chunks"] == 5
    again = indexer.build_index("section")
    assert again["embedded"] == 0 and again["removed"] == 0
    assert dense.count() == 5


def test_rechunking_removes_stale_chunks(env, corpus):
    dense, write, _ = env
    indexer.build_index("section")
    aapl_2025 = [c for c in corpus if c.filing_id == "AAPL-10K-2025FY"]
    write(aapl_2025[:1])  # the filing now has one chunk instead of two
    stats = indexer.build_index("section")
    assert stats["removed"] == 1
    assert retriever.get_store().count("section") == 4
    assert aapl_2025[1].chunk_id not in dense.points
    hits = retriever.get_bm25("section").search("research development", 5)
    assert aapl_2025[1].chunk_id not in {cid for cid, _ in hits}


def test_recreated_collection_triggers_reembedding(env):
    dense, _, _ = env
    indexer.build_index("section")
    dense.points.clear()  # someone wiped Qdrant
    assert indexer.build_index("section")["embedded"] == 5


def test_filing_subset(env):
    stats = indexer.build_index("section", filing_ids={"KO-10K-2025FY"})
    assert stats["loaded"] == 1 and stats["embedded"] == 1


def test_empty_corpus_does_not_crash(tmp_path):
    index = BM25Index.build([])
    assert index.search("anything", 5) == []
    index.save(tmp_path / "bm25")
    assert BM25Index.load(tmp_path / "bm25").search("anything", 5) == []


def test_blank_filters_are_ignored():
    f = SearchFilters(ticker="", form="", doc_name="  ")
    assert f.is_empty()


def test_store_is_safe_across_threads(tmp_path, corpus):
    store = ChunkStore(tmp_path / "chunks.sqlite")
    many = [
        c.model_copy(update={"chunk_id": f"{c.chunk_id}-{i}"}) for i in range(300) for c in corpus
    ]
    store.upsert(many)
    ids = [c.chunk_id for c in many]
    errors: list[Exception] = []
    sizes: list[int] = []

    def reader():
        try:
            for _ in range(5):
                sizes.append(len(store.get_many(ids)))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def writer():
        try:
            for _ in range(5):
                store.upsert(many[:100])
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(8)] + [
        threading.Thread(target=writer)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert set(sizes) == {len(many)}
