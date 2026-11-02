"""One search interface over BM25, dense, hybrid (RRF) and hybrid plus rerank."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from ledgerlens.config import get_settings
from ledgerlens.retrieval.bm25 import BM25Index
from ledgerlens.retrieval.dense import DenseIndex
from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.retrieval.fusion import reciprocal_rank_fusion
from ledgerlens.retrieval.rerank import rerank
from ledgerlens.retrieval.store import ChunkStore
from ledgerlens.schemas import Chunk, ChunkStrategy

Method = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
METHODS: tuple[Method, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


def index_dir() -> Path:
    return get_settings().data_dir / "index"


@lru_cache(maxsize=1)
def get_store() -> ChunkStore:
    return ChunkStore(index_dir() / "chunks.sqlite")


@lru_cache(maxsize=4)
def get_bm25(strategy: ChunkStrategy) -> BM25Index:
    return BM25Index.load(index_dir() / f"bm25_{strategy}")


@lru_cache(maxsize=1)
def _qdrant() -> object:
    from qdrant_client import QdrantClient

    return QdrantClient(url=get_settings().qdrant_url, timeout=60)


def get_dense(strategy: ChunkStrategy) -> DenseIndex:
    from qdrant_client import QdrantClient

    client = _qdrant()
    assert isinstance(client, QdrantClient)
    return DenseIndex(client, strategy)


class Retriever:
    """Search one chunking strategy with one method.

    Hybrid fuses the top `candidates` of BM25 and dense with reciprocal rank fusion.
    Rerank re-scores the fused top `candidates` with a cross-encoder.
    """

    def __init__(
        self,
        strategy: ChunkStrategy = "section",
        method: Method = "hybrid",
        candidates: int = 40,
        rrf_k: int = 60,
    ) -> None:
        self.strategy = strategy
        self.method = method
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 8, filters: SearchFilters | None = None) -> list[Hit]:
        filters = filters or SearchFilters()
        pool = max(k, self.candidates)
        if self.method == "bm25":
            ranked = get_bm25(self.strategy).search(query, k, filters)
        elif self.method == "dense":
            ranked = get_dense(self.strategy).search(query, k, filters)
        else:
            sparse = [cid for cid, _ in get_bm25(self.strategy).search(query, pool, filters)]
            dense = [cid for cid, _ in get_dense(self.strategy).search(query, pool, filters)]
            ranked = reciprocal_rank_fusion([sparse, dense], k=self.rrf_k)[:pool]
            if self.method == "hybrid_rerank":
                chunks = get_store().get_many([cid for cid, _ in ranked])
                pairs = [(cid, chunks[cid].text) for cid, _ in ranked if cid in chunks]
                ranked = rerank(query, pairs, top_k=k)
            ranked = ranked[:k]
        chunks = get_store().get_many([cid for cid, _ in ranked])
        return [
            Hit(chunk=chunks[cid], score=score, rank=i)
            for i, (cid, score) in enumerate(ranked, start=1)
            if cid in chunks
        ]
