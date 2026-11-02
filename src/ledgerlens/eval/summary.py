"""Headline summaries of a run, written into `RunFile.summary` and `summary_by_dataset`."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import numpy as np

from ledgerlens.eval.stats import FloatArray, bootstrap_ci
from ledgerlens.schemas import MetricValue, QuestionResult


def latency_ms(result: QuestionResult) -> float | None:
    """End-to-end latency of one question.

    `wall_ms` when the runner recorded it, else the summed LLM call latency in
    `prediction.usage.latency_ms`, else None (the question crashed before an
    answer existed).
    """
    if result.wall_ms is not None:
        return result.wall_ms
    if result.prediction is not None:
        return result.prediction.usage.latency_ms
    return None


def total_tokens(result: QuestionResult) -> int | None:
    """Prompt plus completion tokens, or None when there is no prediction."""
    if result.prediction is None:
        return None
    usage = result.prediction.usage
    return usage.prompt_tokens + usage.completion_tokens


def cost_usd(result: QuestionResult) -> float | None:
    """List-price-equivalent cost, or None when there is no prediction."""
    return None if result.prediction is None else result.prediction.usage.cost_usd


def _pct(data: FloatArray, q: float) -> float:
    return float(np.percentile(data, q))


def summarize_scores(
    results: Sequence[QuestionResult], n_resamples: int = 2000, seed: int = 0
) -> dict[str, MetricValue]:
    """Bootstrap summary of a run's per-question results.

    * Every key found in any `scores` dict: the mean over questions where it is
      not None, with a 95% percentile-bootstrap interval. Keys with no values
      are left out.
    * `latency_p50_ms`, `latency_p95_ms`: percentiles of `latency_ms`, with
      bootstrap intervals (cheap at eval-set sizes).
    * `tokens_mean`, `cost_usd_mean`: mean tokens and list-price cost per
      answered question, with intervals.
    * `error_rate`: share of questions that crashed, with an interval.

    Keys are sorted so the output is stable. Resampling is seeded, so the same
    results always give the same summary.
    """
    boot = partial(bootstrap_ci, n_resamples=n_resamples, seed=seed)
    summary: dict[str, MetricValue] = {}
    for key in sorted({k for r in results for k in r.scores}):
        values = [r.scores.get(key) for r in results]
        if any(v is not None for v in values):
            summary[key] = boot(values)
    latencies = [latency_ms(r) for r in results]
    if any(v is not None for v in latencies):
        summary["latency_p50_ms"] = boot(latencies, stat=partial(_pct, q=50))
        summary["latency_p95_ms"] = boot(latencies, stat=partial(_pct, q=95))
    tokens = [total_tokens(r) for r in results]
    if any(v is not None for v in tokens):
        summary["tokens_mean"] = boot([None if t is None else float(t) for t in tokens])
        summary["cost_usd_mean"] = boot([cost_usd(r) for r in results])
    if results:
        summary["error_rate"] = boot([1.0 if r.error else 0.0 for r in results])
    return dict(sorted(summary.items()))


def summarize_by_dataset(
    results: Sequence[QuestionResult], n_resamples: int = 2000, seed: int = 0
) -> dict[str, dict[str, MetricValue]]:
    """`summarize_scores` for each dataset in the run, keyed by dataset name."""
    datasets = sorted({r.dataset for r in results})
    return {
        name: summarize_scores(
            [r for r in results if r.dataset == name], n_resamples=n_resamples, seed=seed
        )
        for name in datasets
    }
