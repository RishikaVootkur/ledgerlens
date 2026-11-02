"""Optional cross-encoder rerank on CPU (ONNX via fastembed, no torch)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _encoder() -> Any:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=RERANK_MODEL)


def rerank(query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
    """Score (chunk_id, text) pairs against the query and return the best `top_k`."""
    if not candidates:
        return []
    scores = list(_encoder().rerank(query, [text[:2000] for _, text in candidates]))
    ranked = sorted(
        zip((cid for cid, _ in candidates), scores, strict=True), key=lambda x: -float(x[1])
    )
    return [(cid, float(score)) for cid, score in ranked[:top_k]]
