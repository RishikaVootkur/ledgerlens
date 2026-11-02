"""Build the chunk store, BM25 indexes and Qdrant collections from chunk JSONL files.

Idempotent: the store upserts by chunk id, BM25 is rebuilt from the store (seconds), and
only chunks whose text has not been embedded before are sent to the embedding model.
"""

from __future__ import annotations

from ledgerlens.config import get_settings
from ledgerlens.logging import get_logger
from ledgerlens.retrieval.bm25 import BM25Index
from ledgerlens.retrieval.retriever import get_bm25, get_dense, get_store, index_dir
from ledgerlens.schemas import Chunk, ChunkStrategy

log = get_logger(__name__)


def load_chunk_files(
    strategy: ChunkStrategy, only: set[str] | None = None
) -> dict[str, list[Chunk]]:
    """Chunks per filing id, from DATA_DIR/chunks/{strategy}/{filing_id}.jsonl."""
    folder = get_settings().data_dir / "chunks" / strategy
    out: dict[str, list[Chunk]] = {}
    for path in sorted(folder.glob("*.jsonl")):
        if only is not None and path.stem not in only:
            continue
        with path.open() as fh:
            out[path.stem] = [Chunk.model_validate_json(line) for line in fh if line.strip()]
    return out


def build_index(
    strategy: ChunkStrategy,
    *,
    filing_ids: set[str] | None = None,
    dense: bool = True,
    batch_size: int = 256,
) -> dict[str, int]:
    store = get_store()
    removed: list[str] = []
    loaded = 0
    for filing_id, chunks in load_chunk_files(strategy, filing_ids).items():
        removed.extend(store.replace_filing(filing_id, strategy, chunks))
        loaded += len(chunks)
    all_chunks = list(store.iter_strategy(strategy))
    BM25Index.build(all_chunks).save(index_dir() / f"bm25_{strategy}")
    get_bm25.cache_clear()  # this process must not keep searching the old index
    version_file = index_dir() / f"version_{strategy}.txt"
    # While vectors are being written the index is incomplete: mark it so nothing caches
    # search results under the final fingerprint until embedding has finished.
    version_file.write_text("building")
    stats = {"loaded": loaded, "removed": len(removed), "bm25_chunks": len(all_chunks)}
    stats["embedded"] = 0
    log.info("index.bm25", strategy=strategy, chunks=len(all_chunks), removed=len(removed))
    if not dense:
        version_file.write_text(store.fingerprint(strategy))
        return stats
    index = get_dense(strategy)
    created = index.ensure_collection()
    if created or index.count() < store.embedded_count(strategy):
        # The collection was wiped or recreated: bookkeeping must not claim vectors exist.
        store.reset_embedded(strategy)
    index.delete(removed)
    pending = store.pending_embeddings(strategy)
    if filing_ids is not None:
        pending = [c for c in pending if c.filing_id in filing_ids]
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        index.add(batch)
        store.mark_embedded([c.chunk_id for c in batch])
        stats["embedded"] += len(batch)
        log.info("index.dense", strategy=strategy, done=stats["embedded"], total=len(pending))
    version_file.write_text(store.fingerprint(strategy))
    return stats
