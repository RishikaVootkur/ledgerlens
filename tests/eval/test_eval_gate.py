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
    assert "| numeric_correct | 0.7500 | 0.5000 | -0.2500 | 0.0000 | 4 | 4/4 | FAIL |" in md
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
    assert result.reasons == ["mrr is scored on 0 question(s), down from 1 in the baseline"]
    assert "| mrr | n/a | n/a | n/a | 0.0000 | 0 | 1/0 | FAIL |" in render_gate_markdown(result)


def test_dropping_a_metric_cannot_hide_a_regression(
    make_result: ResultBuilder, make_run: RunBuilder
) -> None:
    # Baseline citation_precision (1, 1, 0) = 0.667. The candidate stops citing on q2,
    # so its precision there becomes None. Over common scored questions (q0, q1) both
    # means are 1.0: the drop check alone would pass. The scored-count check fails it.
    gate = GateThresholds(max_drop={"citation_precision": 0.0}, correctness_metrics=[])
    base, cand = _runs(
        make_result,
        make_run,
        [{"citation_precision": v} for v in (1.0, 1.0, 0.0)],
        [{"citation_precision": v} for v in (1.0, 1.0, None)],
    )
    result = check_gate(base, cand, gate)
    assert not result.passed
    [metric] = result.metrics
    assert (metric.baseline, metric.candidate, metric.n) == (1.0, 1.0, 2)
    assert (metric.n_baseline_scored, metric.n_candidate_scored) == (3, 2)
    assert result.reasons == [
        "citation_precision is scored on 2 question(s), down from 3 in the baseline"
    ]
    # Scoring more questions than the baseline is fine.
    assert check_gate(cand, base, gate).passed


def test_unscored_correctness_counts_as_a_regression(
    make_result: ResultBuilder, make_run: RunBuilder
) -> None:
    base, cand = _runs(make_result, make_run, [{"judge_correct": 1.0}], [{"judge_correct": None}])
    result = check_gate(base, cand, GateThresholds(correctness_metrics=["judge_correct"]))
    [flip] = result.flips
    assert (flip.candidate, flip.direction) == (None, "regressed")
    assert "- `q0` judge_correct: 1 -> not scored" in render_gate_markdown(result)


def test_comment_escapes_user_text(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    base = make_run("base|x", [make_result("q`1|a", {"a|b": 1.0})])
    cand = make_run("cand", [make_result("q`1|a", {"a|b": 0.0}, error="boom")])
    gate = GateThresholds(max_drop={"a|b": 0.0}, correctness_metrics=["a|b"])
    md = render_gate_markdown(check_gate(base, cand, gate))
    assert "Baseline `base\\|x` vs" in md
    assert "| a\\|b | 1.0000 |" in md
    assert "Errored questions: ``q`1\\|a``" in md
    assert "- ``q`1\\|a`` a\\|b: 1 -> 0" in md


def test_ungated_and_default_thresholds(make_result: ResultBuilder, make_run: RunBuilder) -> None:
    base, cand = _runs(make_result, make_run, [{"latency_score": 1.0}], [{"latency_score": 0.0}])
    ungated = check_gate(base, cand, GateThresholds())
    assert ungated.passed
    assert ungated.metrics[0].status == "NOT GATED"
    assert "| latency_score | 1.0000 | 0.0000 | -1.0000 | n/a | 1 | 1/1 | NOT GATED |" in (
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
