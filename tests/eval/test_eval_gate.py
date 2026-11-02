"""CI regression gate decisions and the PR comment."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ledgerlens.eval.gate import (
    COMMENT_MARKER,
    DEFAULT_GATE_FILE,
    GateThresholds,
    check_gate,
    render_gate_markdown,
)
from ledgerlens.schemas import QuestionResult, RunFile

ResultBuilder = Callable[..., QuestionResult]
RunBuilder = Callable[..., RunFile]

STRICT = GateThresholds(max_drop={"numeric_correct": 0.0, "mrr": 0.0})


def _runs(
    make_result: ResultBuilder,
    make_run: RunBuilder,
    base: list[dict[str, float | None]],
    cand: list[dict[str, float | None]],
) -> tuple[RunFile, RunFile]:
    return (
        make_run("base", [make_result(f"q{i}", s) for i, s in enumerate(base)]),
        make_run("cand", [make_result(f"q{i}", s) for i, s in enumerate(cand)]),
    )


def test_identical_runs_pass(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    scores: list[dict[str, float | None]] = [{"numeric_correct": 1.0, "mrr": 0.5}] * 3
    base, cand = _runs(make_result, make_run, scores, scores)
    result = check_gate(base, cand, STRICT)
    assert result.passed
    assert result.reasons == [] and result.flips == []
    assert [m.status for m in result.metrics] == ["PASS", "PASS"]
    md = render_gate_markdown(result)
    assert md.startswith(COMMENT_MARKER)
    assert "## LedgerLens eval gate: PASS" in md
    assert "No question changed correctness." in md
    assert "Failed because" not in md


def test_drop_past_threshold_fails(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    # Baseline numeric_correct (1, 1, 0, 1) = 0.75; candidate (1, 0, 1, 0) = 0.5: drop 0.25.
    base, cand = _runs(
        make_result,
        make_run,
        [{"numeric_correct": v} for v in (1.0, 1.0, 0.0, 1.0)],
        [{"numeric_correct": v} for v in (1.0, 0.0, 1.0, 0.0)],
    )
    result = check_gate(base, cand, STRICT)
    assert not result.passed
    [metric] = result.metrics
    assert (metric.baseline, metric.candidate, metric.delta) == (0.75, 0.5, -0.25)
    assert metric.status == "FAIL"
    assert result.reasons == ["numeric_correct fell by 0.2500, more than the allowed 0.0000"]
    regressed = [f.qid for f in result.flips if f.direction == "regressed"]
    fixed = [f.qid for f in result.flips if f.direction == "fixed"]
    assert (regressed, fixed) == (["q1", "q3"], ["q2"])

    md = render_gate_markdown(result)
    assert "## LedgerLens eval gate: FAIL" in md
    assert "| numeric_correct | 0.7500 | 0.5000 | -0.2500 | 0.0000 | 4 | FAIL |" in md
    assert "Correctness regressed (2):" in md
    assert "- `q1` numeric_correct: 1 -> 0" in md
    assert "Correctness fixed (1):" in md
    # Regressions are listed before fixes.
    assert md.index("regressed") < md.index("fixed")

    # A looser threshold lets the same drop through.
    loose = GateThresholds(max_drop={"numeric_correct": 0.3})
    assert check_gate(base, cand, loose).passed


def test_float_noise_does_not_fail(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    base, cand = _runs(
        make_result, make_run, [{"mrr": 0.1 + 0.2}], [{"mrr": 0.3}]
    )  # 0.30000000000000004 vs 0.3
    assert check_gate(base, cand, STRICT).passed


def test_errored_question_fails(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    base = make_run("base", [make_result("q1", {"numeric_correct": 1.0})])
    cand = make_run("cand", [make_result("q1", {"numeric_correct": 1.0}, error="Timeout")])
    result = check_gate(base, cand, STRICT)
    assert not result.passed
    assert result.errored_qids == ["q1"]
    assert "Errored questions: `q1`" in render_gate_markdown(result)
    allow = GateThresholds(max_drop=STRICT.max_drop, fail_on_error=False)
    assert check_gate(base, cand, allow).passed


def test_missing_question_fails(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    base = make_run("base", [make_result(f"q{i}", {"numeric_correct": 1.0}) for i in range(3)])
    cand = make_run(
        "cand",
        [make_result(f"q{i}", {"numeric_correct": 1.0}) for i in (0, 1, 7)],
    )
    result = check_gate(base, cand, STRICT)
    assert not result.passed
    assert result.missing_qids == ["q2"]
    assert result.n_common == 2  # q7 is new in the candidate and ignored
    assert "Missing from candidate: `q2`" in render_gate_markdown(result)
    allow = GateThresholds(max_drop=STRICT.max_drop, fail_on_missing=False)
    assert check_gate(base, cand, allow).passed


def test_metric_that_disappears_fails(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    base, cand = _runs(make_result, make_run, [{"mrr": 1.0}], [{"mrr": None}])
    result = check_gate(base, cand, STRICT)
    assert not result.passed
    [metric] = result.metrics
    assert (metric.status, metric.candidate, metric.delta, metric.n) == ("FAIL", None, None, 0)
    assert "mrr has baseline values but none in the candidate" in result.reasons
    assert "| mrr | n/a | n/a | n/a | 0.0000 | 0 | FAIL |" in render_gate_markdown(result)


def test_ungated_and_default_thresholds(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    base, cand = _runs(make_result, make_run, [{"latency_score": 1.0}], [{"latency_score": 0.0}])
    ungated = check_gate(base, cand, GateThresholds())
    assert ungated.passed
    assert ungated.metrics[0].status == "NOT GATED"
    assert "| latency_score | 1.0000 | 0.0000 | -1.0000 | n/a | 1 | NOT GATED |" in (
        render_gate_markdown(ungated)
    )
    defaulted = check_gate(base, cand, GateThresholds(default_max_drop=0.5))
    assert not defaulted.passed


def test_flip_cap_in_comment(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    base, cand = _runs(
        make_result,
        make_run,
        [{"numeric_correct": 1.0}] * 5,
        [{"numeric_correct": 0.0}] * 5,
    )
    md = render_gate_markdown(check_gate(base, cand, STRICT), max_flips=2)
    assert "Correctness regressed (5):" in md
    assert "- ... and 3 more" in md


def test_committed_gate_file_loads() -> None:
    thresholds = GateThresholds.from_yaml(DEFAULT_GATE_FILE)
    assert thresholds.threshold_for("numeric_correct") == 0.0
    assert thresholds.threshold_for("not-a-metric") is None
    assert "numeric_correct" in thresholds.correctness_metrics
    assert thresholds.fail_on_error and thresholds.fail_on_missing


def test_gate_file_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text("max_drops:\n  mrr: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_drops"):
        GateThresholds.from_yaml(path)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert GateThresholds.from_yaml(empty) == GateThresholds()
