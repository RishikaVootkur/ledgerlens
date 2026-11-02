"""Build eval/reports/dashboard.json for the eval dashboard (web UI and GitHub Pages).

Which runs appear, and how they are grouped into ablation and comparison tables, is
declared in eval/reports/experiments.yaml, so every number on the dashboard points
back to a committed run file. Deltas and p-values come from paired comparisons
against each table's baseline row (`ledgerlens.eval.compare.compare_runs`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from ledgerlens.config import get_settings
from ledgerlens.eval.compare import compare_runs
from ledgerlens.schemas import RunFile


class RowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    run_id: str


class TableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    description: str
    baseline: str
    columns: list[str]
    rows: list[RowSpec]


class HeadlineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    title: str
    metrics: list[str] | None = None


class ExperimentsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    headline: HeadlineSpec
    tables: list[TableSpec] = []
    failures_from: list[str] = []
    max_failure_examples: int = 40
    judge_calibration: str | None = None


def _runs_dir() -> Path:
    return get_settings().eval_dir / "runs"


def load_run(run_id: str) -> RunFile:
    return RunFile.model_validate_json((_runs_dir() / f"{run_id}.json").read_text())


def _cell(run: RunFile, metric: str, comparison: dict[str, Any] | None) -> dict[str, Any] | None:
    m = run.summary.get(metric)
    if m is None:
        return None
    cell: dict[str, Any] = {"value": m.value, "ci_low": m.ci_low, "ci_high": m.ci_high}
    if comparison is not None:
        cell["delta"] = comparison.get("delta")
        cell["p_value"] = comparison.get("p_value")
    return cell


def build_table(spec: TableSpec) -> dict[str, Any]:
    runs = {row.name: load_run(row.run_id) for row in spec.rows}
    if spec.baseline not in runs:
        raise ValueError(f"table {spec.id}: baseline {spec.baseline!r} is not one of its rows")
    base = runs[spec.baseline]
    rows = []
    for row in spec.rows:
        run = runs[row.name]
        paired: dict[str, dict[str, Any]] = {}
        if row.name != spec.baseline:
            report = compare_runs(base, run)
            paired = {m.name: {"delta": m.delta, "p_value": m.p_value} for m in report.metrics}
        cells: dict[str, Any] = {}
        for metric in spec.columns:
            comparison = paired.get(metric)
            if comparison is None and row.name != spec.baseline:
                # Operational metrics (latency, cost) have no per-question paired test:
                # report the plain difference of the summaries.
                a, b = base.summary.get(metric), run.summary.get(metric)
                comparison = {"delta": b.value - a.value if a and b else None, "p_value": None}
            cell = _cell(run, metric, comparison)
            if cell is not None:
                cells[metric] = cell
        rows.append({"name": row.name, "run_id": row.run_id, "metrics": cells})
    return {
        "id": spec.id,
        "title": spec.title,
        "description": spec.description,
        "baseline": spec.baseline,
        "columns": spec.columns,
        "rows": rows,
    }


def _failures(run_ids: list[str], limit: int) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for run_id in run_ids:
        run = load_run(run_id)
        for r in run.results:
            if r.failure_category is None:
                continue
            counts[r.failure_category] = counts.get(r.failure_category, 0) + 1
            examples.append(
                {
                    "qid": r.qid,
                    "dataset": r.dataset,
                    "question": r.question,
                    "gold": r.gold,
                    "answer": r.prediction.answer if r.prediction else None,
                    "failure_category": r.failure_category,
                    "run_id": run.run_id,
                    "explanation": r.failure_note,
                }
            )
    # Round-robin across categories so the list is not dominated by the largest one.
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for e in sorted(examples, key=lambda e: e["qid"]):
        by_cat.setdefault(e["failure_category"], []).append(e)
    spread: list[dict[str, Any]] = []
    while any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat]:
                spread.append(by_cat[cat].pop(0))
    examples = spread
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])), examples[:limit]


def build_dashboard(spec: ExperimentsSpec) -> dict[str, Any]:
    headline = load_run(spec.headline.run_id)
    metrics = spec.headline.metrics or list(headline.summary)
    referenced = {spec.headline.run_id, *spec.failures_from}
    referenced |= {row.run_id for table in spec.tables for row in table.rows}
    runs = sorted((load_run(r) for r in referenced), key=lambda r: r.created_at, reverse=True)
    failure_counts, failures = _failures(spec.failures_from, spec.max_failure_examples)
    calibration = None
    if spec.judge_calibration:
        data = json.loads((get_settings().eval_dir.parent / spec.judge_calibration).read_text())
        calibration = {k: v for k, v in data.items() if k != "results"}
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "headline": {
            "run_id": headline.run_id,
            "title": spec.headline.title,
            "metrics": {
                m: headline.summary[m].model_dump() for m in metrics if m in headline.summary
            },
        },
        "runs": [
            {
                "run_id": r.run_id,
                "config_name": r.config_name,
                "split": r.split,
                "created_at": r.created_at.isoformat(),
                "datasets": list(r.datasets),
                "summary": {k: v.model_dump() for k, v in r.summary.items()},
            }
            for r in runs
        ],
        "tables": [build_table(t) for t in spec.tables],
        "failure_counts": failure_counts,
        "failures": failures,
        "judge_calibration": calibration,
    }


def write_dashboard(spec_path: Path, out: Path | None = None) -> tuple[Path, list[str]]:
    """Write dashboard.json and return it with the run ids it references (for publishing)."""
    spec = ExperimentsSpec.model_validate(yaml.safe_load(spec_path.read_text()))
    data = build_dashboard(spec)
    out = out or get_settings().eval_dir / "reports" / "dashboard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    return out, [r["run_id"] for r in data["runs"]]
