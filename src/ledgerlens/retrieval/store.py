"""SQLite chunk store: the source of truth for chunk text and metadata by id.

Qdrant and BM25 return chunk ids; everything that shows a passage (citations, the
`get_table` tool, the UI's source panel) reads it from here. SQLite keeps memory flat
in the API container no matter how large the corpus grows.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

from ledgerlens.schemas import Chunk, ChunkStrategy

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    filing_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    body TEXT NOT NULL,
    body_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_strategy ON chunks(strategy, seq);
CREATE INDEX IF NOT EXISTS ix_chunks_filing ON chunks(filing_id, strategy);
CREATE TABLE IF NOT EXISTS embedded (
    chunk_id TEXT PRIMARY KEY,
    body_hash TEXT NOT NULL
);
"""


class ChunkStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)

    def upsert(self, chunks: Iterable[Chunk]) -> int:
        rows = [
            (c.chunk_id, c.strategy, c.filing_id, _seq(c.chunk_id), body, _hash(body))
            for c in chunks
            for body in (c.model_dump_json(),)
        ]
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO chunks"
                "(chunk_id, strategy, filing_id, seq, body, body_hash) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def get(self, chunk_id: str) -> Chunk | None:
        row = self._db.execute("SELECT body FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return Chunk.model_validate_json(row[0]) if row else None

    def get_many(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        out: dict[str, Chunk] = {}
        for i in range(0, len(chunk_ids), 500):
            batch = chunk_ids[i : i + 500]
            marks = ",".join("?" * len(batch))
            for (body,) in self._db.execute(
                f"SELECT body FROM chunks WHERE chunk_id IN ({marks})", batch
            ):
                chunk = Chunk.model_validate_json(body)
                out[chunk.chunk_id] = chunk
        return out

    def iter_strategy(self, strategy: ChunkStrategy) -> Iterator[Chunk]:
        """All chunks of one strategy in a stable order (filing, then sequence)."""
        cursor = self._db.execute(
            "SELECT body FROM chunks WHERE strategy = ? ORDER BY filing_id, seq", (strategy,)
        )
        for (body,) in cursor:
            yield Chunk.model_validate_json(body)

    def filing_chunks(self, filing_id: str, strategy: ChunkStrategy) -> list[Chunk]:
        cursor = self._db.execute(
            "SELECT body FROM chunks WHERE filing_id = ? AND strategy = ? ORDER BY seq",
            (filing_id, strategy),
        )
        return [Chunk.model_validate_json(body) for (body,) in cursor]

    def pending_embeddings(self, strategy: ChunkStrategy) -> list[Chunk]:
        """Chunks whose current text has not been embedded yet (resumable indexing)."""
        cursor = self._db.execute(
            "SELECT c.body FROM chunks c LEFT JOIN embedded e ON e.chunk_id = c.chunk_id "
            "AND e.body_hash = c.body_hash WHERE c.strategy = ? AND e.chunk_id IS NULL "
            "ORDER BY c.filing_id, c.seq",
            (strategy,),
        )
        return [Chunk.model_validate_json(body) for (body,) in cursor]

    def mark_embedded(self, chunk_ids: list[str]) -> None:
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO embedded(chunk_id, body_hash) "
                "SELECT chunk_id, body_hash FROM chunks WHERE chunk_id = ?",
                [(cid,) for cid in chunk_ids],
            )

    def count(self, strategy: ChunkStrategy | None = None) -> int:
        if strategy is None:
            return int(self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        query = "SELECT COUNT(*) FROM chunks WHERE strategy = ?"
        return int(self._db.execute(query, (strategy,)).fetchone()[0])

    def filings(self) -> list[dict[str, object]]:
        """One metadata row per filing (from its first chunk), for listings and filters."""
        rows = self._db.execute(
            "SELECT body FROM chunks WHERE seq = (SELECT MIN(seq) FROM chunks c2 "
            "WHERE c2.filing_id = chunks.filing_id AND c2.strategy = chunks.strategy) "
            "AND strategy = 'section'"
        )
        keys = (
            "filing_id",
            "ticker",
            "company",
            "form",
            "fiscal_year",
            "fiscal_period",
            "doc_name",
        )
        out = []
        for (body,) in rows:
            data = json.loads(body)
            out.append({k: data.get(k) for k in keys})
        return out


def _seq(chunk_id: str) -> int:
    tail = chunk_id.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()[:16]
