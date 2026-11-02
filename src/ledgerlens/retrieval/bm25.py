"""BM25 keyword index (bm25s) with exact metadata filtering.

Filtering scores the whole corpus and masks it, instead of retrieving a top-N and
filtering afterwards, so a narrow filter (one filing) never comes back short.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import bm25s
import numpy as np

from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.schemas import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOP = frozenset(
    {
        *("a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have"),
        *("in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was"),
        *("were", "will", "with", "which", "what", "how", "did", "does", "do"),
    }
)
_META_KEYS = ("ticker", "fiscal_year", "form", "filing_id", "doc_name")


def tokenize(text: str) -> list[str]:
    """Lowercase word and number tokens. Numbers keep their separators ("1,234.5")."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, chunk_ids: list[str], meta: dict[str, list[object]]):
        self.retriever = retriever
        self.chunk_ids = chunk_ids
        self.meta = {k: np.asarray(v, dtype=object) for k, v in meta.items()}

    @classmethod
    def build(cls, chunks: list[Chunk]) -> BM25Index:
        retriever = bm25s.BM25(k1=1.2, b=0.75)
        retriever.index([tokenize(c.text) for c in chunks], show_progress=False)
        meta = {k: [getattr(c, k) for c in chunks] for k in _META_KEYS}
        return cls(retriever, [c.chunk_id for c in chunks], meta)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(path / "bm25"))
        meta = {k: v.tolist() for k, v in self.meta.items()}
        (path / "ids.json").write_text(json.dumps({"chunk_ids": self.chunk_ids, "meta": meta}))

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        retriever = bm25s.BM25.load(str(path / "bm25"))
        data = json.loads((path / "ids.json").read_text())
        return cls(retriever, data["chunk_ids"], data["meta"])

    def _mask(self, filters: SearchFilters) -> np.ndarray | None:
        if filters.is_empty():
            return None
        mask = np.ones(len(self.chunk_ids), dtype=bool)
        for key, value in filters.model_dump().items():
            if value is not None:
                expected = value.upper() if key == "ticker" else value
                mask &= self.meta[key] == expected
        return mask

    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens or not self.chunk_ids:
            return []
        scores = np.asarray(self.retriever.get_scores(tokens), dtype=float)
        mask = self._mask(filters or SearchFilters())
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]
