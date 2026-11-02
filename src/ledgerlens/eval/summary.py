"""Headline summaries of a run, written into `RunFile.summary` and `summary_by_dataset`."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Literal

import numpy as np

from ledgerlens.eval.stats import FloatArray, bootstrap_ci, is_binary, wilson_ci
from ledgerlens.schemas import MetricValue, QuestionResult

LatencySource = Literal["wall_ms", "usage"]


def latency_source(results: Sequence[QuestionResult]) -> LatencySource:
    """Which latency to use for a set of results: one source for every row.

    `wall_ms` (end-to-end time) when every result has it. Otherwise the summed
    LLM call latency in `prediction.usage.latency_ms` for all of them. Mixing
    the two in one percentile would compare different things, and summed call
    latency is near 0 on a cache replay.
    """
    if results and all(r.wall_ms is not None for r in results):
        return "wall_ms"
    return "usage"


def latencies_ms(
    results: Sequence[QuestionResult], source: LatencySource | None = None
) -> list[float | None]:
    """Per-question latency from one source (chosen by `latency_source` when not given).

    None for a question with no value from that source, such as a crash before
    an answer existed.
    """
    source = source or latency_source(results)
    if source == "wall_ms":
        return [r.wall_ms for r in results]
    return [None if r.prediction is None else r.prediction.usage.latency_ms for r in results]


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
    """Summary of a run's per-question results, each with a 95% interval.

    * Every key found in any `scores` dict: the mean over questions where it is
      not None. Binary keys (every value 0 or 1) get a Wilson score interval,
      which stays meaningful at 0% and 100% on small slices; continuous keys
      get a percentile bootstrap interval. Keys with no values are left out.
    * `latency_p50_ms`, `latency_p95_ms`: percentiles of one latency source
      (see `latency_source`), with bootstrap intervals.
    * `model_latency_p50_ms`, `model_latency_p95_ms`: percentiles of summed LLM call
      latency, which excludes rate-limit waits.
    * `tokens_mean`, `cost_usd_mean`: mean tokens and list-price cost per
      answered question, with bootstrap intervals.
    * `error_rate`: share of questions that crashed, with a Wilson interval.

    Keys are sorted so the output is stable. Resampling is seeded, so the same
    results always give the same summary.
    """
    boot = partial(bootstrap_ci, n_resamples=n_resamples, seed=seed)
    summary: dict[str, MetricValue] = {}
    for key in sorted({k for r in results for k in r.scores}):
        values = [r.scores.get(key) for r in results]
        if is_binary(values):
            summary[key] = wilson_ci(values)
        elif any(v is not None for v in values):
            summary[key] = boot(values)
    latencies = latencies_ms(results)
    if any(v is not None for v in latencies):
        summary["latency_p50_ms"] = boot(latencies, stat=partial(_pct, q=50))
        summary["latency_p95_ms"] = boot(latencies, stat=partial(_pct, q=95))
    # Model time alone (summed LLM call latency). Wall time on free tiers is dominated by
    # rate-limit pacing, so model latency is the fairer number for comparing models.
    model_ms = [r.prediction.usage.latency_ms if r.prediction else None for r in results]
    if any(v for v in model_ms):
        summary["model_latency_p50_ms"] = boot(model_ms, stat=partial(_pct, q=50))
        summary["model_latency_p95_ms"] = boot(model_ms, stat=partial(_pct, q=95))
    tokens = [total_tokens(r) for r in results]
    if any(v is not None for v in tokens):
        summary["tokens_mean"] = boot([None if t is None else float(t) for t in tokens])
        summary["cost_usd_mean"] = boot([cost_usd(r) for r in results])
    if results:
        summary["error_rate"] = wilson_ci([1.0 if r.error else 0.0 for r in results])
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
