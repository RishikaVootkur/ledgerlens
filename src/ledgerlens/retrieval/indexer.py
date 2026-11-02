"""Build the chunk store, BM25 indexes and Qdrant collections from chunk JSONL files.

Idempotent: the store upserts by chunk id, BM25 is rebuilt from the store (seconds), and
only chunks whose text has not been embedded before are sent to the embedding model.
"""

from __future__ import annotations

from pathlib import Path

from ledgerlens.config import get_settings
from ledgerlens.logging import get_logger
from ledgerlens.retrieval.bm25 import BM25Index
from ledgerlens.retrieval.retriever import get_dense, get_store, index_dir
from ledgerlens.schemas import Chunk, ChunkStrategy

log = get_logger(__name__)


def load_chunk_files(strategy: ChunkStrategy, only: set[str] | None = None) -> list[Chunk]:
    folder = get_settings().data_dir / "chunks" / strategy
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.jsonl")):
        if only is not None and path.stem not in only:
            continue
        chunks.extend(Chunk.model_validate_json(line) for line in path.open() if line.strip())
    return chunks


def build_index(
    strategy: ChunkStrategy,
    *,
    filing_ids: set[str] | None = None,
    dense: bool = True,
    batch_size: int = 256,
) -> dict[str, int]:
    store = get_store()
    loaded = store.upsert(load_chunk_files(strategy, filing_ids))
    all_chunks = list(store.iter_strategy(strategy))
    BM25Index.build(all_chunks).save(index_dir() / f"bm25_{strategy}")
    stats = {"loaded": loaded, "bm25_chunks": len(all_chunks), "embedded": 0}
    log.info("index.bm25", strategy=strategy, chunks=len(all_chunks))
    if not dense:
        return stats
    index = get_dense(strategy)
    index.ensure_collection()
    pending = store.pending_embeddings(strategy)
    if filing_ids is not None:
        pending = [c for c in pending if c.filing_id in filing_ids]
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        index.add(batch)
        store.mark_embedded([c.chunk_id for c in batch])
        stats["embedded"] += len(batch)
        log.info("index.dense", strategy=strategy, done=stats["embedded"], total=len(pending))
    return stats


def bm25_path(strategy: ChunkStrategy) -> Path:
    return index_dir() / f"bm25_{strategy}"
