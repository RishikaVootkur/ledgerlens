"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, 2009)."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum_i w_i / (k + rank_i(d)), ranks starting at 1.

    Ties break by best single rank, then id, so the output is deterministic.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))
