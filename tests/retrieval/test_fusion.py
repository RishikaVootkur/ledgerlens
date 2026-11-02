import pytest

from ledgerlens.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_hand_computed():
    # k = 60. A: rank 1 in list 1, rank 3 in list 2 -> 1/61 + 1/63 = 0.032266
    #         B: rank 2 in list 1, rank 1 in list 2 -> 1/62 + 1/61 = 0.032522
    #         C: rank 3 in list 1 only              -> 1/63       = 0.015873
    #         D: rank 2 in list 2 only              -> 1/62       = 0.016129
    fused = reciprocal_rank_fusion([["A", "B", "C"], ["B", "D", "A"]], k=60)
    assert [d for d, _ in fused] == ["B", "A", "D", "C"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1][1] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_ignores_duplicates_and_breaks_ties_deterministically():
    fused = reciprocal_rank_fusion([["A", "A", "B"], ["B", "A"]], k=60)
    # Ranks are positions in each list; a repeated id keeps its first rank only.
    # A: rank 1 in list 1 and rank 2 in list 2 -> 1/61 + 1/62
    # B: rank 3 in list 1 and rank 1 in list 2 -> 1/63 + 1/61
    scores = dict(fused)
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["B"] == pytest.approx(1 / 63 + 1 / 61)
    assert [d for d, _ in reciprocal_rank_fusion([["X"], ["Y"]])] == ["X", "Y"]


def test_rrf_weights():
    fused = reciprocal_rank_fusion([["A"], ["B"]], k=60, weights=[1.0, 2.0])
    assert fused[0][0] == "B"
