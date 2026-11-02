"""Paired run comparison and its rendering."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ledgerlens.eval.compare import (
    ComparisonReport,
    align,
    compare_runs,
    render_markdown,
    render_text,
    significance_marker,
)
from ledgerlens.schemas import QuestionResult, RunFile

ResultBuilder = Callable[..., QuestionResult]
RunBuilder = Callable[..., RunFile]


@pytest.fixture
def report(make_result: ResultBuilder, make_run: RunBuilder) -> ComparisonReport:
    # A has q1..q4, B has q2..q5: three questions in common.
    a = make_run(
        "run-a",
        [
            make_result("q1", {"numeric_correct": 0.0}, latency_ms=999),
            make_result("q2", {"numeric_correct": 1.0, "mrr": 0.5}, latency_ms=100),
            make_result("q3", {"numeric_correct": 0.0, "mrr": 1.0}, latency_ms=200),
            make_result("q4", {"numeric_correct": 1.0, "mrr": None}, latency_ms=300),
        ],
    )
    b = make_run(
        "run-b",
        [
            make_result("q2", {"numeric_correct": 1.0, "mrr": 1.0}, latency_ms=50, cost=0.0),
            make_result("q3", {"numeric_correct": 1.0, "mrr": 0.5}, latency_ms=150, cost=0.0),
            make_result("q4", {"numeric_correct": 1.0, "mrr": 0.25}, latency_ms=250, cost=0.0),
            make_result("q5", {"numeric_correct": 0.0}, latency_ms=999),
        ],
    )
    return compare_runs(a, b)


def test_alignment_with_mismatched_qids(report: ComparisonReport) -> None:
    assert report.n_common == 3
    assert report.only_in_a == ["q1"]
    assert report.only_in_b == ["q5"]
    assert (report.run_a, report.run_b) == ("run-a", "run-b")
    assert (report.config_a, report.config_b) == ("cfg-run-a", "cfg-run-b")


def test_binary_metric_uses_mcnemar(report: ComparisonReport) -> None:
    m = {x.name: x for x in report.metrics}["numeric_correct"]
    # Common questions q2..q4: A = (1, 0, 1) -> 2/3, B = (1, 1, 1) -> 1, delta = 1/3.
    assert m.test == "mcnemar"
    assert m.n == 3
    assert m.mean_a == pytest.approx(2 / 3)
    assert m.mean_b == 1.0
    assert m.delta == pytest.approx(1 / 3)
    # One discordant pair (q3, B right): p = 2 * P(X <= 0 | n=1) capped at 1.
    assert m.p_value == 1.0
    assert m.ci_low <= m.delta <= m.ci_high


def test_continuous_metric_uses_permutation(report: ComparisonReport) -> None:
    m = {x.name: x for x in report.metrics}["mrr"]
    # q4 has no A value, so pairs are q2 (0.5 -> 1.0) and q3 (1.0 -> 0.5).
    # Differences +0.5 and -0.5: mean 0. Every sign flip has |mean| >= 0, so p = 1.
    assert m.test == "permutation"
    assert m.n == 2
    assert m.mean_a == 0.75 and m.mean_b == 0.75
    assert m.delta == 0.0
    assert m.p_value == 1.0
    assert m.p_holm == 1.0


def test_operational_metrics_over_common_questions(report: ComparisonReport) -> None:
    ops = {o.name: o for o in report.operational}
    # Latency of q2..q4 only: A (100, 200, 300), B (50, 150, 250).
    assert ops["latency_p50_ms"].a == 200.0 and ops["latency_p50_ms"].b == 150.0
    assert ops["latency_p50_ms"].delta == -50.0
    # p95 of (100, 200, 300) = 200 + 0.9 * 100 = 290.
    assert ops["latency_p95_ms"].a == pytest.approx(290.0)
    # 3 questions x 110 tokens each.
    assert ops["total_tokens"].a == 330.0
    assert ops["total_cost_usd"].a == pytest.approx(0.003)
    assert ops["total_cost_usd"].b == 0.0
    assert ops["errors"].a == 0.0


def test_holm_adjustment_and_markers(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    # 12 questions where B fixes every one of A's misses: McNemar n01 = 0, n10 = 12,
    # p = 2 / 2^12 = 0.000488. A second metric with p = 1 doubles nothing past 1.
    a = make_run("a", [make_result(f"q{i}", {"x": 0.0, "y": 0.5}) for i in range(12)])
    b = make_run("b", [make_result(f"q{i}", {"x": 1.0, "y": 0.5}) for i in range(12)])
    report = compare_runs(a, b)
    x = {m.name: m for m in report.metrics}["x"]
    assert x.p_value == pytest.approx(2 / 4096)
    # Holm with m = 2: smallest p is multiplied by 2.
    assert x.p_holm == pytest.approx(4 / 4096)
    md = render_markdown(report)
    assert "+1.000***" in md
    assert "<0.001" in md


def test_significance_marker() -> None:
    assert significance_marker(0.0005) == "***"
    assert significance_marker(0.005) == "**"
    assert significance_marker(0.03) == "*"
    assert significance_marker(0.05) == ""


def test_render_markdown(report: ComparisonReport) -> None:
    md = render_markdown(report)
    lines = md.splitlines()
    assert lines[0] == "## Run comparison: `run-a` (A) vs `run-b` (B)"
    assert "3 questions in both runs; dropped 1 only in A and 1 only in B." in md
    assert "| Metric | A | B | Delta (B-A) | 95% CI | p | p (Holm) | Test | n |" in lines
    numeric_row = next(line for line in lines if line.startswith("| numeric_correct |"))
    assert numeric_row.startswith("| numeric_correct | 0.667 | 1.000 | +0.333 | [")
    assert numeric_row.endswith("| 1.000 | 1.000 | mcnemar | 3 |")
    assert "| latency_p50_ms | 200 | 150 | -50 |" in lines
    assert "| total_tokens | 330 | 330 | +0 |" in lines
    assert "Holm-adjusted" in md


def test_render_text_is_aligned(report: ComparisonReport) -> None:
    text = render_text(report)
    lines = text.splitlines()
    assert lines[0] == "A: run-a (cfg-run-a)"
    header = next(line for line in lines if line.startswith("Metric"))
    rule = lines[lines.index(header) + 1]
    assert set(rule.replace(" ", "")) == {"-"}
    rows = [line for line in lines if line.startswith(("mrr", "numeric_correct"))]
    # Right-aligned columns end at the same place in every row.
    assert len({len(r) for r in [header, *rows]}) == 1


def test_render_without_shared_metrics(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    a = make_run("a", [make_result("q1", {"x": None})])
    b = make_run("b", [make_result("q1", {"x": 1.0}, error="boom", answered=False)])
    report = compare_runs(a, b)
    assert report.metrics == []
    ops = {o.name: o for o in report.operational}
    assert ops["latency_p50_ms"].b is None and ops["latency_p50_ms"].delta is None
    assert ops["errors"].b == 1.0
    for text in (render_markdown(report), render_text(report)):
        assert "No per-question metric has values in both runs." in text
        assert "n/a" in text


def test_number_formatting(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    a = make_run("a", [make_result("q1", {}, cost=0.00002, tokens=(5000, 0))])
    b = make_run("b", [make_result("q1", {}, cost=0.00001, tokens=(7000, 0))])
    md = render_markdown(compare_runs(a, b))
    assert "| total_cost_usd | 2.00e-05 | 1.00e-05 | -1.00e-05 |" in md
    assert "| total_tokens | 5,000 | 7,000 | +2,000 |" in md
    c = make_run("c", [make_result("q1", {}, latency_ms=1234.56, cost=0.25)])
    md = render_markdown(compare_runs(c, c))
    assert "| latency_p50_ms | 1,234.6 | 1,234.6 | +0 |" in md
    assert "| total_cost_usd | 0.2500 | 0.2500 | +0 |" in md


def test_duplicate_qids_are_rejected(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    dup = make_run("dup", [make_result("q1", {}), make_result("q1", {})])
    with pytest.raises(ValueError, match="duplicate qid q1"):
        align(dup, dup)
