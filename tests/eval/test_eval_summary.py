"""Run summaries written into RunFile.summary and summary_by_dataset."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ledgerlens.eval.summary import (
    cost_usd,
    latency_ms,
    summarize_by_dataset,
    summarize_scores,
    total_tokens,
)
from ledgerlens.schemas import QuestionResult

ResultBuilder = Callable[..., QuestionResult]


def test_summarize_scores_bootstraps_each_key(make_result: ResultBuilder) -> None:
    results = [
        make_result("q1", {"numeric_correct": 1.0, "mrr": 0.5}, latency_ms=100, cost=0.002),
        make_result("q2", {"numeric_correct": 0.0, "mrr": None}, latency_ms=200, cost=0.004),
        make_result("q3", {"numeric_correct": 1.0, "unused": None}, latency_ms=300, cost=0.006),
    ]
    summary = summarize_scores(results, n_resamples=200)
    assert list(summary) == sorted(summary)
    # numeric_correct: (1 + 0 + 1) / 3 over 3 questions.
    assert summary["numeric_correct"].value == pytest.approx(2 / 3)
    assert summary["numeric_correct"].n == 3
    # mrr skips the None: one value.
    assert summary["mrr"].value == 0.5 and summary["mrr"].n == 1
    # A key with no values at all is left out.
    assert "unused" not in summary
    assert summary["latency_p50_ms"].value == 200.0
    # p95 of (100, 200, 300): position 2 * 0.95 = 1.9 -> 200 + 0.9 * 100 = 290.
    assert summary["latency_p95_ms"].value == pytest.approx(290.0)
    # Default tokens are 100 prompt + 10 completion.
    assert summary["tokens_mean"].value == 110.0
    assert summary["cost_usd_mean"].value == pytest.approx(0.004)
    assert summary["error_rate"].value == 0.0
    ci = summary["numeric_correct"]
    assert ci.ci_low is not None and ci.ci_high is not None
    assert ci.ci_low <= ci.value <= ci.ci_high


def test_summary_handles_errors_and_wall_time(make_result: ResultBuilder) -> None:
    ok = make_result("q1", {"numeric_correct": 1.0}, latency_ms=50, wall_ms=900)
    crashed = make_result("q2", {"numeric_correct": None}, error="boom", answered=False)
    assert latency_ms(ok) == 900  # wall time wins over summed call latency
    assert latency_ms(crashed) is None
    assert total_tokens(crashed) is None and cost_usd(crashed) is None
    summary = summarize_scores([ok, crashed], n_resamples=50)
    assert summary["latency_p50_ms"].value == 900
    assert summary["latency_p50_ms"].n == 1
    assert summary["error_rate"].value == 0.5


def test_summary_of_nothing(make_result: ResultBuilder) -> None:
    assert summarize_scores([]) == {}
    only_crashes = [make_result("q1", {}, error="boom", answered=False)]
    assert set(summarize_scores(only_crashes, n_resamples=10)) == {"error_rate"}


def test_summarize_by_dataset(make_result: ResultBuilder) -> None:
    results = [
        make_result("x1", {"numeric_correct": 1.0}, dataset="xbrl"),
        make_result("x2", {"numeric_correct": 0.0}, dataset="xbrl"),
        make_result("u1", {"refusal_correct": 1.0}, dataset="unanswerable"),
    ]
    by_ds = summarize_by_dataset(results, n_resamples=50)
    assert list(by_ds) == ["unanswerable", "xbrl"]
    assert by_ds["xbrl"]["numeric_correct"].value == 0.5
    assert "numeric_correct" not in by_ds["unanswerable"]
    assert by_ds["unanswerable"]["refusal_correct"].n == 1
