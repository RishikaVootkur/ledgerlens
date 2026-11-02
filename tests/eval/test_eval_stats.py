"""Statistics against worked examples and simple invariants."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom  # type: ignore[import-untyped]

from ledgerlens.eval.stats import (
    agreement_accuracy,
    bootstrap_ci,
    cohen_kappa,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_diff,
    paired_permutation_test,
    percentile,
    summarize_latency,
)

# ---------------------------------------------------------------- bootstrap


def test_bootstrap_is_deterministic_with_seed() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    first = bootstrap_ci(values, seed=7)
    assert bootstrap_ci(values, seed=7) == first
    # A different seed draws different resamples (visible on continuous data).
    smooth = [0.11, 0.52, 0.93, 0.24, 0.65, 0.36, 0.87, 0.08, 0.49, 0.7]
    assert bootstrap_ci(smooth, seed=7) != bootstrap_ci(smooth, seed=8)
    assert first.value == 0.7
    assert first.n == 10
    assert first.ci_low is not None and first.ci_high is not None
    assert first.ci_low <= 0.7 <= first.ci_high


def test_bootstrap_drops_none() -> None:
    result = bootstrap_ci([1.0, None, 3.0, None])
    assert result.value == 2.0
    assert result.n == 2


def test_bootstrap_single_value_and_constant_collapse() -> None:
    one = bootstrap_ci([0.4])
    assert (one.value, one.ci_low, one.ci_high, one.n) == (0.4, 0.4, 0.4, 1)
    flat = bootstrap_ci([2.0] * 20)
    assert flat.ci_low == pytest.approx(2.0) and flat.ci_high == pytest.approx(2.0)


def test_bootstrap_custom_statistic() -> None:
    result = bootstrap_ci([1.0, 2.0, 3.0, 100.0], stat=np.median)
    assert result.value == 2.5


def test_bootstrap_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_ci([None, None])


def test_bootstrap_coverage_is_near_nominal() -> None:
    # 200 samples of n=60 from N(0, 1): the 95% interval should contain the true mean 0
    # roughly 95% of the time. Percentile intervals undercover slightly at small n,
    # so accept 88% to 99%.
    rng = np.random.default_rng(123)
    hits = 0
    trials = 200
    for i in range(trials):
        sample = rng.normal(0.0, 1.0, size=60).tolist()
        ci = bootstrap_ci(sample, n_resamples=400, seed=i)
        assert ci.ci_low is not None and ci.ci_high is not None
        hits += ci.ci_low <= 0.0 <= ci.ci_high
    assert 0.88 <= hits / trials <= 0.99


# ---------------------------------------------------------------- paired bootstrap


def test_paired_bootstrap_identical_inputs() -> None:
    a = [0.2, 0.5, 0.9, 0.1]
    assert paired_bootstrap_diff(a, a) == (0.0, 0.0, 0.0, 1.0)


def test_paired_bootstrap_constant_shift() -> None:
    # Every pair differs by exactly 1, so every resample does too: the CI is [1, 1]
    # and no centered resample reaches |1|, so p = (1 + 0) / (2000 + 1).
    b = [0.1, 0.4, 0.3, 0.8, 0.2]
    a = [x + 1 for x in b]
    diff, low, high, p = paired_bootstrap_diff(a, b)
    assert diff == pytest.approx(1.0)
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)
    assert p == pytest.approx(1 / 2001)


def test_paired_bootstrap_drops_incomplete_pairs_and_is_seeded() -> None:
    a = [1.0, None, 0.0, 1.0, 1.0, 0.0]
    b = [0.0, 1.0, None, 1.0, 0.0, 0.0]
    # Complete pairs: (1,0), (1,1), (1,0), (0,0) -> differences 1, 0, 1, 0 -> mean 0.5.
    diff, low, high, p = paired_bootstrap_diff(a, b, seed=3)
    assert diff == 0.5
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert 0.0 < p <= 1.0
    assert paired_bootstrap_diff(a, b, seed=3) == (diff, low, high, p)


def test_paired_bootstrap_errors() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="complete pair"):
        paired_bootstrap_diff([None], [1.0])


# ---------------------------------------------------------------- McNemar


def test_mcnemar_worked_example() -> None:
    # 20 questions. 6 both right, 4 both wrong (concordant, ignored).
    # Discordant: A right / B wrong n01 = 1, B right / A wrong n10 = 9, n = 10.
    # Under H0, n01 ~ Binomial(10, 0.5). Two-sided exact p:
    #   P(X <= 1) = (C(10,0) + C(10,1)) / 2^10 = (1 + 10) / 1024 = 11/1024
    #   p = 2 * 11/1024 = 22/1024 = 0.021484375
    a = [True] * 6 + [False] * 4 + [True] * 1 + [False] * 9
    b = [True] * 6 + [False] * 4 + [False] * 1 + [True] * 9
    assert mcnemar_exact(a, b) == pytest.approx(22 / 1024)
    # The test is symmetric in A and B.
    assert mcnemar_exact(b, a) == pytest.approx(22 / 1024)


def test_mcnemar_second_worked_example() -> None:
    # Second worked example: discordant counts b = 5 and c = 15, n = 20.
    # Exact p from binomial coefficients C(20, 0..5):
    #   P(X <= 5) = (1 + 20 + 190 + 1140 + 4845 + 15504) / 2^20 = 21700 / 1048576
    #   p = 2 * 21700 / 1048576 = 0.041389
    a = [True] * 5 + [False] * 15
    b = [False] * 5 + [True] * 15
    assert mcnemar_exact(a, b) == pytest.approx(43400 / 1048576)


def test_mcnemar_wikipedia_example() -> None:
    # The example table in Wikipedia's "McNemar's test" article: a = 101, b = 121,
    # c = 59, d = 33. Its chi-squared statistic is (121 - 59)^2 / (121 + 59) = 21.36,
    # p < 0.001. The exact version is 2 * P(X <= 59) for X ~ Binomial(180, 0.5).
    a = [True] * 101 + [True] * 121 + [False] * 59 + [False] * 33
    b = [True] * 101 + [False] * 121 + [True] * 59 + [False] * 33
    p = mcnemar_exact(a, b)
    assert p == pytest.approx(2 * binom.cdf(59, 180, 0.5))
    assert p < 0.001


def test_mcnemar_no_discordant_pairs() -> None:
    assert mcnemar_exact([True, False, True], [True, False, True]) == 1.0
    # Balanced disagreement: n01 = n10 = 2 -> p = 1.
    assert mcnemar_exact([True, True, False, False], [False, False, True, True]) == 1.0


def test_mcnemar_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        mcnemar_exact([True], [True, False])


# ---------------------------------------------------------------- permutation


def test_permutation_identical_inputs_p_is_one() -> None:
    a = [0.3, 0.6, 0.9]
    assert paired_permutation_test(a, a) == 1.0
    # Large n uses the random branch; still p = 1 when every difference is 0.
    big = list(np.linspace(0, 1, 40))
    assert paired_permutation_test(big, big) == 1.0


def test_permutation_exact_enumeration_hand_computed() -> None:
    # Differences d = (1, 2, 3), observed |mean| = 2 (sum 6).
    # All 2^3 sign flips give sums 6, 4, 2, 0, 0, -2, -4, -6.
    # |sum| >= 6 in 2 of 8 -> p = 0.25.
    assert paired_permutation_test([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 0.25


def test_permutation_random_branch_detects_shift_and_is_seeded() -> None:
    rng = np.random.default_rng(0)
    b = rng.normal(0, 1, size=50).tolist()
    a = [x + 0.8 + rng.normal(0, 0.3) for x in b]
    p = paired_permutation_test(a, b, n_resamples=2000, seed=1)
    # No sign flip reaches the observed mean, so p is the floor 1 / 2001.
    assert p == pytest.approx(1 / 2001)
    assert paired_permutation_test(a, b, n_resamples=2000, seed=1) == p


def test_permutation_drops_none_and_handles_empty() -> None:
    assert paired_permutation_test([None, 1.0], [1.0, None]) == 1.0
    assert paired_permutation_test([1.0, None, 2.0, 3.0], [0.0, 5.0, 0.0, 0.0]) == 0.25


# ---------------------------------------------------------------- agreement


def test_cohen_kappa_classic_example() -> None:
    # 50 items rated yes/no by two raters:
    #                 B yes   B no
    #     A yes        20       5     (A says yes 25 times)
    #     A no         10      15     (A says no 25 times)
    # po = (20 + 15) / 50 = 0.70
    # pe = P(A yes) P(B yes) + P(A no) P(B no) = 0.5 * 0.6 + 0.5 * 0.4 = 0.50
    # kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert agreement_accuracy(a, b) == pytest.approx(0.70)
    assert cohen_kappa(a, b) == pytest.approx(0.40)


def test_cohen_kappa_edges() -> None:
    # Perfect disagreement on balanced labels: po = 0, pe = 0.5 -> kappa = -1.
    assert cohen_kappa([1, 0], [0, 1]) == pytest.approx(-1.0)
    assert cohen_kappa([1, 0, 2], [1, 0, 2]) == pytest.approx(1.0)
    # Degenerate: both raters always say True, pe = 1 -> defined as 1.0.
    assert cohen_kappa([True, True, True], [True, True, True]) == 1.0
    with pytest.raises(ValueError, match="differ in length"):
        cohen_kappa([1], [1, 0])
    with pytest.raises(ValueError, match="at least one"):
        agreement_accuracy([], [])


# ---------------------------------------------------------------- latency


def test_percentile_linear_interpolation() -> None:
    # [1, 2, 3, 4]: p50 position (n-1) * 0.5 = 1.5 -> 2.5.
    # p95 position 3 * 0.95 = 2.85 -> 3 + 0.85 * (4 - 3) = 3.85.
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], 101)


def test_summarize_latency() -> None:
    summary = summarize_latency([100.0, None, 200.0, 300.0, 400.0])
    assert summary == {
        "p50": 250.0,
        "p95": pytest.approx(385.0),  # position 3 * 0.95 = 2.85 -> 300 + 0.85 * 100
        "mean": 250.0,
        "max": 400.0,
        "n": 4.0,
    }
    assert summarize_latency([None]) == {"n": 0.0}


# ---------------------------------------------------------------- Holm


def test_holm_bonferroni_hand_computed() -> None:
    # p = (0.01, 0.04, 0.03, 0.005), m = 4. Ascending with multipliers m - rank:
    #   0.005 * 4 = 0.02, 0.01 * 3 = 0.03, 0.03 * 2 = 0.06, 0.04 * 1 = 0.04
    # Running max: 0.02, 0.03, 0.06, 0.06. Back in input order:
    assert holm_bonferroni([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    # Capped at 1.
    assert holm_bonferroni([0.5, 0.9]) == [1.0, 1.0]
    assert holm_bonferroni([]) == []
