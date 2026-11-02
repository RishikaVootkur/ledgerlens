"""CI regression gate: fail a PR when the candidate run is worse than the baseline.

CI replays the committed response cache, so the replayed slice is deterministic:
any drop in a quality metric is a real behavior change, not sampling noise. The
gate compares the two runs question by question, applies per-metric thresholds
from `eval/gate.yaml`, and renders the PR comment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.config import ROOT
from ledgerlens.eval.compare import align
from ledgerlens.eval.markdown import md_cell, md_code
from ledgerlens.schemas import RunFile

DEFAULT_GATE_FILE = ROOT / "eval" / "gate.yaml"
COMMENT_MARKER = "<!-- ledgerlens-eval-gate -->"
"""Hidden marker so the CI job can find and update its own PR comment."""

_EPS = 1e-9
"""Float slack: a drop of 1e-12 on a 0.0 threshold is rounding, not a regression."""

Status = Literal["PASS", "FAIL", "NOT GATED"]


class GateThresholds(BaseModel):
    """Gate settings. Every score key is treated as higher-is-better."""

    model_config = ConfigDict(extra="forbid")

    max_drop: dict[str, float] = Field(
        default_factory=dict, description="Largest allowed fall in the mean, per metric"
    )
    default_max_drop: float | None = Field(
        default=None, description="Threshold for metrics not in max_drop; None leaves them ungated"
    )
    correctness_metrics: list[str] = Field(
        default_factory=lambda: ["numeric_correct", "refusal_correct"],
        description="Score keys whose per-question flips are listed in the PR comment",
    )
    fail_on_error: bool = True
    fail_on_missing: bool = True

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_GATE_FILE) -> GateThresholds:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def threshold_for(self, metric: str) -> float | None:
        return self.max_drop.get(metric, self.default_max_drop)


class GateMetric(BaseModel):
    name: str
    baseline: float | None
    candidate: float | None
    n: int = Field(description="Questions both runs scored; the means use these")
    n_baseline_scored: int = 0
    n_candidate_scored: int = 0
    threshold: float | None
    status: Status

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline


class Flip(BaseModel):
    """A question whose correctness changed between baseline and candidate."""

    qid: str
    metric: str
    baseline: float
    candidate: float | None = Field(description="None when the candidate stopped scoring it")

    @property
    def direction(self) -> Literal["regressed", "fixed"]:
        if self.candidate is None:
            return "regressed"
        return "regressed" if self.candidate < self.baseline else "fixed"


class GateResult(BaseModel):
    passed: bool
    baseline_run: str
    candidate_run: str
    n_common: int
    metrics: list[GateMetric]
    errored_qids: list[str]
    missing_qids: list[str]
    flips: list[Flip]
    reasons: list[str]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def check_gate(baseline: RunFile, candidate: RunFile, thresholds: GateThresholds) -> GateResult:
    """Decide whether `candidate` may merge given `baseline`.

    Means are compared over the questions both runs scored (both values not
    None), so a metric moves only because answers changed. The gate fails when:

    * any gated metric's mean falls by more than its threshold (baseline minus
      candidate > threshold);
    * the candidate scores a gated metric on fewer common questions than the
      baseline. Many metrics are None when they do not apply (no citations,
      judge failure), so without this check a candidate could hide its worst
      answers by no longer producing the metric for them;
    * any candidate question errored (`fail_on_error`);
    * any baseline question is missing from the candidate (`fail_on_missing`).

    Metrics without a threshold are reported as NOT GATED. Questions only in the
    candidate are ignored (new questions have no baseline to regress from).
    """
    pairs, only_baseline, _ = align(baseline, candidate)
    reasons: list[str] = []
    metrics: list[GateMetric] = []
    names = sorted(
        {k for b, _ in pairs for k in b.scores} | {k for _, c in pairs for k in c.scores}
    )
    for name in names:
        kept = [
            (vb, vc)
            for b, c in pairs
            if (vb := b.scores.get(name)) is not None and (vc := c.scores.get(name)) is not None
        ]
        n_base = sum(1 for b, _ in pairs if b.scores.get(name) is not None)
        n_cand = sum(1 for _, c in pairs if c.scores.get(name) is not None)
        threshold = thresholds.threshold_for(name)
        base_mean = _mean([v for v, _ in kept]) if kept else None
        cand_mean = _mean([v for _, v in kept]) if kept else None
        status: Status = "NOT GATED" if threshold is None else "PASS"
        if threshold is not None:
            if base_mean is not None and cand_mean is not None:
                drop = base_mean - cand_mean
                if drop > threshold + _EPS:
                    status = "FAIL"
                    reasons.append(
                        f"{name} fell by {drop:.4f}, more than the allowed {threshold:.4f}"
                    )
            if n_cand < n_base:
                status = "FAIL"
                reasons.append(
                    f"{name} is scored on {n_cand} question(s), down from {n_base} in the baseline"
                )
        metrics.append(
            GateMetric(
                name=name,
                baseline=base_mean,
                candidate=cand_mean,
                n=len(kept),
                n_baseline_scored=n_base,
                n_candidate_scored=n_cand,
                threshold=threshold,
                status=status,
            )
        )
    errored = sorted(c.qid for _, c in pairs if c.error)
    if thresholds.fail_on_error and errored:
        reasons.append(f"{len(errored)} candidate question(s) errored")
    if thresholds.fail_on_missing and only_baseline:
        reasons.append(f"{len(only_baseline)} baseline question(s) missing from the candidate")
    flips = [
        Flip(qid=b.qid, metric=name, baseline=vb, candidate=vc)
        for b, c in pairs
        for name in thresholds.correctness_metrics
        if (vb := b.scores.get(name)) is not None and (vc := c.scores.get(name)) != vb
    ]
    return GateResult(
        passed=not reasons,
        baseline_run=baseline.run_id,
        candidate_run=candidate.run_id,
        n_common=len(pairs),
        metrics=metrics,
        errored_qids=errored,
        missing_qids=only_baseline,
        flips=flips,
        reasons=reasons,
    )


def _codes(qids: list[str]) -> str:
    return ", ".join(md_code(q) for q in qids)


def _fmt(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def render_gate_markdown(result: GateResult, max_flips: int = 50) -> str:
    """PR comment body: verdict, reasons, metric diff table and flipped questions.

    Flips are grouped into regressions first, then fixes, each capped at
    `max_flips` lines so a broad change cannot produce an unreadable comment.
    """
    verdict = "PASS" if result.passed else "FAIL"
    lines = [
        COMMENT_MARKER,
        f"## LedgerLens eval gate: {verdict}",
        "",
        f"Baseline {md_code(result.baseline_run)} vs candidate {md_code(result.candidate_run)} "
        f"on {result.n_common} replayed questions.",
        "",
    ]
    if result.reasons:
        lines += ["Failed because:", "", *(f"- {md_cell(r)}" for r in result.reasons), ""]
    lines += [
        "| Metric | Baseline | Candidate | Delta | Max drop | n | Scored (base/cand) | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {md_cell(m.name)} | {_fmt(m.baseline)} | {_fmt(m.candidate)} "
        f"| {_fmt(m.delta, signed=True)} | {_fmt(m.threshold)} | {m.n} "
        f"| {m.n_baseline_scored}/{m.n_candidate_scored} | {m.status} |"
        for m in result.metrics
    ]
    lines.append("")
    if result.errored_qids:
        lines += [f"Errored questions: {_codes(result.errored_qids)}", ""]
    if result.missing_qids:
        lines += [f"Missing from candidate: {_codes(result.missing_qids)}", ""]
    if not result.flips:
        lines += ["No question changed correctness.", ""]
    for direction in ("regressed", "fixed"):
        group = [f for f in result.flips if f.direction == direction]
        if not group:
            continue
        lines += [f"Correctness {direction} ({len(group)}):", ""]
        lines += [
            f"- {md_code(f.qid)} {md_cell(f.metric)}: {f.baseline:g} -> "
            + ("not scored" if f.candidate is None else f"{f.candidate:g}")
            for f in group[:max_flips]
        ]
        if len(group) > max_flips:
            lines.append(f"- ... and {len(group) - max_flips} more")
        lines.append("")
    return "\n".join(lines)
