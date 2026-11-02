"""Statistics for the eval lab: confidence intervals, paired tests and agreement.

Every resampling function takes a seed and uses its own `numpy.random.Generator`,
so the same inputs always give the same interval or p-value. Inputs may contain
None (a metric that did not apply to a question); those entries are dropped, and
for paired functions the whole pair is dropped.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats as sps

from ledgerlens.schemas import MetricValue

FloatArray = npt.NDArray[np.float64]
Statistic = Callable[[FloatArray], Any]
"""A statistic of a 1-D array that returns a scalar, such as `np.mean` or `np.median`."""


def _clean(values: Sequence[float | None]) -> FloatArray:
    return np.asarray([v for v in values if v is not None], dtype=np.float64)


def _pairs(a: Sequence[float | None], b: Sequence[float | None]) -> tuple[FloatArray, FloatArray]:
    if len(a) != len(b):
        raise ValueError(f"paired inputs must have the same length, got {len(a)} and {len(b)}")
    kept = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    return (
        np.asarray([x for x, _ in kept], dtype=np.float64),
        np.asarray([y for _, y in kept], dtype=np.float64),
    )


def _resample(data: FloatArray, stat: Statistic, n_resamples: int, seed: int) -> FloatArray:
    """`stat` of `n_resamples` bootstrap resamples (with replacement) of `data`."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(data), size=(n_resamples, len(data)))
    return np.asarray([float(stat(data[row])) for row in idx], dtype=np.float64)


def bootstrap_ci(
    values: Sequence[float | None],
    stat: Statistic = np.mean,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> MetricValue:
    """Point estimate and percentile-bootstrap confidence interval.

    None values are dropped and `n` is the number of values used. The interval
    is the (alpha/2, 1 - alpha/2) percentiles of `stat` over `n_resamples`
    resamples, so the default is a 95% interval. With a single value the
    interval collapses to that value. Raises ValueError when no values remain,
    because there is nothing to estimate.
    """
    data = _clean(values)
    if data.size == 0:
        raise ValueError("bootstrap_ci needs at least one non-None value")
    boot = _resample(data, stat, n_resamples, seed)
    low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return MetricValue(
        value=float(stat(data)), ci_low=float(low), ci_high=float(high), n=int(data.size)
    )


def paired_bootstrap_diff(
    a: Sequence[float | None],
    b: Sequence[float | None],
    stat: Statistic = np.mean,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Difference `stat(a - b)` over aligned pairs, with a bootstrap CI and p-value.

    Pairs where either side is None are dropped. Questions are resampled as
    pairs, which keeps the per-question correlation between the two systems
    and gives a much tighter interval than two independent bootstraps.

    Returns `(diff, ci_low, ci_high, p_value)`. The interval uses the percentile
    method. The two-sided p-value tests "no difference" by centering the
    bootstrap distribution on zero (shifting it by the observed difference) and
    counting resamples at least as far from zero as the observed difference:
    p = (1 + #{|d* - d| >= |d|}) / (n_resamples + 1). The +1 keeps p above zero,
    as a finite resample cannot show a p-value of exactly 0.
    """
    xa, xb = _pairs(a, b)
    if xa.size == 0:
        raise ValueError("paired_bootstrap_diff needs at least one complete pair")
    diffs = xa - xb
    observed = float(stat(diffs))
    boot = _resample(diffs, stat, n_resamples, seed)
    low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    extreme = np.abs(boot - observed) >= abs(observed) - _eps(observed)
    p_value = (1 + int(extreme.sum())) / (n_resamples + 1)
    return observed, float(low), float(high), min(1.0, p_value)


def _eps(x: float) -> float:
    """Float slack for "at least as extreme" comparisons."""
    return 1e-12 * max(1.0, abs(x))


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Two-sided exact McNemar test for paired binary outcomes.

    Only discordant pairs carry information: n01 counts questions A got right
    and B got wrong, n10 the reverse. Under the null both kinds are equally
    likely, so n01 ~ Binomial(n01 + n10, 0.5) and the p-value is the exact
    two-sided binomial test. Returns 1.0 when there are no discordant pairs.
    """
    if len(a) != len(b):
        raise ValueError(f"paired inputs must have the same length, got {len(a)} and {len(b)}")
    n01 = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    n10 = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    if n01 + n10 == 0:
        return 1.0
    return float(sps.binomtest(n01, n01 + n10, 0.5, alternative="two-sided").pvalue)


def paired_permutation_test(
    a: Sequence[float | None],
    b: Sequence[float | None],
    n_resamples: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation (sign-flip) test on the mean difference.

    Under the null that A and B are exchangeable within each question, every
    per-question difference d_i = a_i - b_i is equally likely to have either
    sign. The p-value is the share of sign assignments whose |mean| is at least
    the observed |mean(d)|. When 2^n <= `n_resamples` all assignments are
    enumerated and the p-value is exact; otherwise `n_resamples` random
    assignments are drawn and p = (1 + #extreme) / (n_resamples + 1).
    Pairs with a None are dropped; with no pairs left the result is 1.0.
    """
    xa, xb = _pairs(a, b)
    diffs = xa - xb
    n = diffs.size
    if n == 0:
        return 1.0
    observed = abs(float(diffs.mean()))
    threshold = observed - _eps(observed)
    if 2**n <= n_resamples:
        signs = np.asarray(list(itertools.product((1.0, -1.0), repeat=n)), dtype=np.float64)
        means = np.abs((signs * diffs).mean(axis=1))
        return float((means >= threshold).sum() / len(signs))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([1.0, -1.0]), size=(n_resamples, n))
    means = np.abs((signs * diffs).mean(axis=1))
    return min(1.0, (1 + int((means >= threshold).sum())) / (n_resamples + 1))


def agreement_accuracy(y1: Sequence[Hashable], y2: Sequence[Hashable]) -> float:
    """Observed agreement: the share of items where the two raters give the same label."""
    _check_raters(y1, y2)
    return sum(1 for x, y in zip(y1, y2, strict=True) if x == y) / len(y1)


def cohen_kappa(y1: Sequence[Hashable], y2: Sequence[Hashable]) -> float:
    """Cohen's kappa for two raters over binary or categorical labels.

    kappa = (po - pe) / (1 - pe), where po is the observed agreement and pe the
    agreement expected by chance from each rater's label frequencies,
    pe = sum over labels c of p1(c) * p2(c).

    Degenerate case: pe == 1 only when both raters give every item the same
    single label, so they also agree on every item (po == 1). Kappa is 0/0
    there; this returns 1.0 because the raters agree perfectly, and callers
    should report the raw agreement next to it.
    """
    po = agreement_accuracy(y1, y2)
    n = len(y1)
    labels = set(y1) | set(y2)
    pe = sum((list(y1).count(c) / n) * (list(y2).count(c) / n) for c in labels)
    if np.isclose(pe, 1.0):
        return 1.0
    return (po - pe) / (1 - pe)


def _check_raters(y1: Sequence[Hashable], y2: Sequence[Hashable]) -> None:
    if len(y1) != len(y2):
        raise ValueError(f"rater label lists differ in length: {len(y1)} and {len(y2)}")
    if not y1:
        raise ValueError("need at least one rated item")


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0 to 100) with numpy's linear interpolation."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be between 0 and 100, got {q}")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_latency(latencies_ms: Sequence[float | None]) -> dict[str, float]:
    """p50, p95, mean and max latency in milliseconds, plus the count `n`.

    None values (questions without a timing) are dropped. Returns {"n": 0} for
    no timings.
    """
    data = [v for v in latencies_ms if v is not None]
    if not data:
        return {"n": 0.0}
    return {
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "mean": float(np.mean(data)),
        "max": float(np.max(data)),
        "n": float(len(data)),
    }


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Sort the m p-values ascending; the i-th smallest (0-based) is multiplied by
    (m - i), then a running maximum keeps the adjusted values monotone, and each
    is capped at 1. Rejecting where the adjusted p < alpha controls the
    family-wise error rate at alpha across all m comparisons, which is what an
    ablation table with many rows needs.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted
