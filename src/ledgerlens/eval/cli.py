"""`ledgerlens eval ...`: run, compare, gate and retrieval-only evaluation."""

from __future__ import annotations

from pathlib import Path

import typer

from ledgerlens.schemas import RunFile, Split

eval_app = typer.Typer(help="Evaluation lab: run experiments, compare runs, gate regressions.")


def _split(value: str) -> Split:
    if value not in ("dev", "test"):
        raise typer.BadParameter("split must be dev or test")
    return value  # type: ignore[return-value]


def _load_run(ref: str) -> RunFile:
    from ledgerlens.config import get_settings

    path = Path(ref)
    if not path.exists():
        path = get_settings().eval_dir / "runs" / f"{ref}.json"
    return RunFile.model_validate_json(path.read_text())


@eval_app.command("run")
def run(
    config: Path = typer.Option(..., help="Eval config YAML"),
    split: str = typer.Option("dev", help="dev or test (test is held out: report once)"),
    limit: int | None = typer.Option(None, help="Only the first N questions"),
    out: Path | None = typer.Option(None, help="Run file path (default eval/runs/<id>.json)"),
    run_id: str | None = typer.Option(None, help="Run id (default timestamp-config-split)"),
    record_cache: bool = typer.Option(False, help="Copy touched responses into eval/cache"),
) -> None:
    """Run the agent over a split and write a run file."""
    from ledgerlens.eval.runner import EvalConfig, run_eval, write_run
    from ledgerlens.eval.runner import record_cache as do_record

    cfg = EvalConfig.from_yaml(config)
    if limit is not None:
        cfg = cfg.model_copy(update={"limit": limit})

    def progress(done: int, total: int, result: object) -> None:
        r = result
        mark = "ERR" if getattr(r, "error", None) else "ok "
        typer.echo(f"[{done}/{total}] {mark} {getattr(r, 'qid', '')}", err=True)

    run_file = run_eval(cfg, _split(split), run_id=run_id, progress=progress)
    path = write_run(run_file, out)
    typer.echo(f"wrote {path}")
    for name, m in run_file.summary.items():
        ci = f" [{m.ci_low:.3f}, {m.ci_high:.3f}]" if m.ci_low is not None else ""
        typer.echo(f"  {name:<22} {m.value:10.4f}{ci}  n={m.n}")
    if record_cache:
        typer.echo(f"recorded {do_record()} cache entries into eval/cache")


@eval_app.command("retrieval")
def retrieval(
    config: Path = typer.Option(..., help="Eval config YAML (strategy, method, datasets)"),
    split: str = typer.Option("dev"),
    out: Path | None = typer.Option(None),
    run_id: str | None = typer.Option(None),
) -> None:
    """Score retrieval alone (no LLM) against labeled evidence."""
    from ledgerlens.eval.retrieval_eval import run_retrieval_eval
    from ledgerlens.eval.runner import EvalConfig, write_run

    run_file = run_retrieval_eval(EvalConfig.from_yaml(config), _split(split), run_id=run_id)
    path = write_run(run_file, out)
    typer.echo(f"wrote {path}")
    for name, m in run_file.summary.items():
        ci = f" [{m.ci_low:.3f}, {m.ci_high:.3f}]" if m.ci_low is not None else ""
        typer.echo(f"  {name:<22} {m.value:10.4f}{ci}  n={m.n}")


@eval_app.command("compare")
def compare(
    run_a: str = typer.Argument(..., help="Baseline run id or path"),
    run_b: str = typer.Argument(..., help="Candidate run id or path"),
    markdown: bool = typer.Option(False, help="Print Markdown instead of text"),
) -> None:
    """Paired comparison of two runs on their common questions, with significance."""
    from ledgerlens.eval.compare import compare_runs, render_markdown, render_text

    report = compare_runs(_load_run(run_a), _load_run(run_b))
    typer.echo(render_markdown(report) if markdown else render_text(report))


@eval_app.command("gate")
def gate(
    baseline: str = typer.Option(..., help="Baseline run id or path"),
    candidate: str = typer.Option(..., help="Candidate run id or path"),
    thresholds: Path = typer.Option(Path("eval/gate.yaml")),
    markdown_out: Path | None = typer.Option(None, help="Write the PR comment body here"),
) -> None:
    """Fail (exit 1) if any gated metric regressed past its threshold."""
    from ledgerlens.eval.gate import GateThresholds, check_gate, render_gate_markdown

    result = check_gate(
        _load_run(baseline), _load_run(candidate), GateThresholds.from_yaml(thresholds)
    )
    body = render_gate_markdown(result)
    if markdown_out:
        markdown_out.write_text(body)
    typer.echo(body)
    raise typer.Exit(0 if result.passed else 1)


@eval_app.command("judge-calibrate")
def judge_calibrate(
    split: str = typer.Option("test", help="dev for prompt iteration, test to report"),
    model: str | None = typer.Option(None, help="Judge model alias (default GPT-OSS 120B)"),
    out: Path | None = typer.Option(None, help="Report path (default eval/reports/...)"),
    record_cache: bool = typer.Option(False, help="Copy touched responses into eval/cache"),
) -> None:
    """Measure the LLM judge against FinanceBench's human labels."""
    from ledgerlens.config import get_settings
    from ledgerlens.eval.calibration import calibrate
    from ledgerlens.eval.runner import record_cache as do_record

    report = calibrate(_split(split), model)
    path = (
        out
        or get_settings().eval_dir
        / "reports"
        / f"judge_calibration_{report.judge_model}_{split}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=1))
    typer.echo(f"wrote {path}")
    for name in ("accuracy", "kappa", "error_rate", "binary_accuracy", "binary_kappa"):
        m = getattr(report, name)
        typer.echo(f"  {name:<16} {m.value:.3f} [{m.ci_low:.3f}, {m.ci_high:.3f}] n={m.n}")
    typer.echo(f"  confusion (rows human, cols judge, {report.labels}): {report.confusion}")
    if record_cache:
        typer.echo(f"recorded {do_record()} cache entries into eval/cache")
