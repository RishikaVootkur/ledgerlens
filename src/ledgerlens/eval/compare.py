"""Paired comparison of two eval runs, as printed by `ledgerlens eval compare`.

Runs are aligned by question id, so every test is paired: each question is
scored by both systems, and the per-question differences carry far more signal
than two independent averages.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from ledgerlens.eval.stats import (
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_diff,
    paired_permutation_test,
    percentile,
)
from ledgerlens.eval.summary import cost_usd, latency_ms, total_tokens
from ledgerlens.schemas import QuestionResult, RunFile

TestName = Literal["mcnemar", "permutation"]


class MetricComparison(BaseModel):
    """One quality metric compared across runs. Delta is B minus A."""

    name: str
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    p_holm: float
    test: TestName
    n: int


class OperationalComparison(BaseModel):
    """A cost or speed figure for each run over the common questions. None if not measurable."""

    name: str
    a: float | None
    b: float | None

    @property
    def delta(self) -> float | None:
        return None if self.a is None or self.b is None else self.b - self.a


class ComparisonReport(BaseModel):
    run_a: str
    run_b: str
    config_a: str
    config_b: str
    n_common: int
    only_in_a: list[str]
    only_in_b: list[str]
    metrics: list[MetricComparison]
    operational: list[OperationalComparison]


def _by_qid(run: RunFile) -> dict[str, QuestionResult]:
    out: dict[str, QuestionResult] = {}
    for result in run.results:
        if result.qid in out:
            raise ValueError(f"run {run.run_id} has duplicate qid {result.qid}")
        out[result.qid] = result
    return out


def align(
    a: RunFile, b: RunFile
) -> tuple[list[tuple[QuestionResult, QuestionResult]], list[str], list[str]]:
    """Pair results by qid in run A's order.

    Returns the pairs plus the qids found only in A and only in B (each sorted).
    Duplicate qids within a run are an error, because pairing would be ambiguous.
    """
    ra, rb = _by_qid(a), _by_qid(b)
    pairs = [(ra[q], rb[q]) for q in ra if q in rb]
    return pairs, sorted(set(ra) - set(rb)), sorted(set(rb) - set(ra))


def _is_binary(values: Sequence[float]) -> bool:
    return all(v in (0.0, 1.0) for v in values)


def _compare_metric(
    name: str, pairs: list[tuple[QuestionResult, QuestionResult]], seed: int
) -> MetricComparison | None:
    kept = [
        (va, vb)
        for ra, rb in pairs
        if (va := ra.scores.get(name)) is not None and (vb := rb.scores.get(name)) is not None
    ]
    if not kept:
        return None
    xa = [va for va, _ in kept]
    xb = [vb for _, vb in kept]
    delta, low, high, _ = paired_bootstrap_diff(xb, xa, seed=seed)
    test: TestName
    if _is_binary(xa + xb):
        test, p_value = "mcnemar", mcnemar_exact([v == 1.0 for v in xa], [v == 1.0 for v in xb])
    else:
        test, p_value = "permutation", paired_permutation_test(xb, xa, seed=seed)
    return MetricComparison(
        name=name,
        mean_a=sum(xa) / len(xa),
        mean_b=sum(xb) / len(xb),
        delta=delta,
        ci_low=low,
        ci_high=high,
        p_value=p_value,
        p_holm=p_value,
        test=test,
        n=len(kept),
    )


def _operational(results: list[QuestionResult]) -> dict[str, float | None]:
    latencies = [v for r in results if (v := latency_ms(r)) is not None]
    tokens = [v for r in results if (v := total_tokens(r)) is not None]
    costs = [v for r in results if (v := cost_usd(r)) is not None]
    return {
        "latency_p50_ms": percentile(latencies, 50) if latencies else None,
        "latency_p95_ms": percentile(latencies, 95) if latencies else None,
        "total_tokens": float(sum(tokens)) if tokens else None,
        "total_cost_usd": float(sum(costs)) if costs else None,
        "errors": float(sum(1 for r in results if r.error)),
    }


def compare_runs(a: RunFile, b: RunFile, seed: int = 0) -> ComparisonReport:
    """Compare run B against run A on the questions both runs contain.

    For each per-question score key present in either run, over the questions
    where both runs have a value:

    * mean A, mean B and delta = mean B - mean A;
    * a 95% paired-bootstrap interval for the delta;
    * a two-sided p-value: exact McNemar when every value is 0 or 1, else a
      paired sign-flip permutation test on the mean difference;
    * a Holm-Bonferroni adjusted p-value across all metrics in the report,
      since a table of many metrics will otherwise show false positives.

    Operational figures (latency p50/p95, total tokens, total list-price cost,
    error count) are computed over the same common questions. Questions only
    in one run are dropped and listed in the report.
    """
    pairs, only_a, only_b = align(a, b)
    names = sorted({k for ra, rb in pairs for k in (*ra.scores, *rb.scores)})
    metrics = [m for name in names if (m := _compare_metric(name, pairs, seed)) is not None]
    for metric, adjusted in zip(
        metrics, holm_bonferroni([m.p_value for m in metrics]), strict=True
    ):
        metric.p_holm = adjusted
    op_a = _operational([ra for ra, _ in pairs])
    op_b = _operational([rb for _, rb in pairs])
    return ComparisonReport(
        run_a=a.run_id,
        run_b=b.run_id,
        config_a=a.config_name,
        config_b=b.config_name,
        n_common=len(pairs),
        only_in_a=only_a,
        only_in_b=only_b,
        metrics=metrics,
        operational=[OperationalComparison(name=k, a=op_a[k], b=op_b[k]) for k in op_a],
    )


# ---------------------------------------------------------------- rendering


def significance_marker(p: float) -> str:
    """`***` for p < 0.001, `**` for p < 0.01, `*` for p < 0.05, else empty."""
    for cutoff, marker in ((0.001, "***"), (0.01, "**"), (0.05, "*")):
        if p < cutoff:
            return marker
    return ""


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_num(x: float, signed: bool = False) -> str:
    """Quality metrics: three decimals, float noise below 1e-12 shown as zero."""
    if abs(x) < 1e-12:
        x = 0.0
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def _fmt_op(x: float | None, signed: bool = False) -> str:
    """Operational figures: whole numbers with separators, small costs in scientific form."""
    if x is None:
        return "n/a"
    sign = "+" if signed else ""
    if abs(x) < 1e-12:
        return f"{0:{sign}.0f}"
    if x == int(x):
        return f"{x:{sign},.0f}"
    if abs(x) >= 100:
        return f"{x:{sign},.1f}"
    if abs(x) < 0.01:
        return f"{x:{sign}.2e}"
    return f"{x:{sign}.4f}"


def _metric_rows(report: ComparisonReport) -> list[list[str]]:
    return [
        [
            m.name,
            _fmt_num(m.mean_a),
            _fmt_num(m.mean_b),
            _fmt_num(m.delta, signed=True) + significance_marker(m.p_holm),
            f"[{_fmt_num(m.ci_low, signed=True)}, {_fmt_num(m.ci_high, signed=True)}]",
            _fmt_p(m.p_value),
            _fmt_p(m.p_holm),
            m.test,
            str(m.n),
        ]
        for m in report.metrics
    ]


def _op_rows(report: ComparisonReport) -> list[list[str]]:
    return [
        [o.name, _fmt_op(o.a), _fmt_op(o.b), _fmt_op(o.delta, signed=True)]
        for o in report.operational
    ]


_METRIC_HEADER = ["Metric", "A", "B", "Delta (B-A)", "95% CI", "p", "p (Holm)", "Test", "n"]
_OP_HEADER = ["Operational", "A", "B", "Delta (B-A)"]
_FOOTNOTE = (
    "Significance markers use Holm-adjusted p-values: * p<0.05, ** p<0.01, *** p<0.001. "
    "CI is a 95% paired bootstrap interval of the delta."
)


def _coverage_line(report: ComparisonReport) -> str:
    return (
        f"{report.n_common} questions in both runs; dropped {len(report.only_in_a)} only in A "
        f"and {len(report.only_in_b)} only in B."
    )


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def render_markdown(report: ComparisonReport) -> str:
    """The report as GitHub Markdown: a header, a metric table and an operational table."""
    lines = [
        f"## Run comparison: `{report.run_a}` (A) vs `{report.run_b}` (B)",
        "",
        f"Configs: A = `{report.config_a}`, B = `{report.config_b}`. {_coverage_line(report)}",
        "",
    ]
    if report.metrics:
        lines += [*_md_table(_METRIC_HEADER, _metric_rows(report)), ""]
    else:
        lines += ["No per-question metric has values in both runs.", ""]
    lines += [*_md_table(_OP_HEADER, _op_rows(report)), "", _FOOTNOTE, ""]
    return "\n".join(lines)


def _text_table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        return "  ".join(
            c.ljust(w) if i == 0 else c.rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        ).rstrip()

    return [line(header), "  ".join("-" * w for w in widths), *(line(r) for r in rows)]


def render_text(report: ComparisonReport) -> str:
    """The report as an aligned plain-text table for the terminal."""
    lines = [
        f"A: {report.run_a} ({report.config_a})",
        f"B: {report.run_b} ({report.config_b})",
        _coverage_line(report),
        "",
    ]
    if report.metrics:
        lines += [*_text_table(_METRIC_HEADER, _metric_rows(report)), ""]
    else:
        lines += ["No per-question metric has values in both runs.", ""]
    lines += [*_text_table(_OP_HEADER, _op_rows(report)), "", _FOOTNOTE, ""]
    return "\n".join(lines)
