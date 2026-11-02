"""`ledgerlens` command line entry point."""

from __future__ import annotations

import typer

from ledgerlens import __version__
from ledgerlens.agent.cli import ask
from ledgerlens.config import get_settings
from ledgerlens.eval.cli import eval_app
from ledgerlens.logging import configure_logging
from ledgerlens.mcp_server import mcp_app
from ledgerlens.retrieval.cli import index_app

app = typer.Typer(help="LedgerLens: cited answers from SEC filings, with an eval lab.")
app.add_typer(index_app, name="index")
app.command("ask")(ask)
app.add_typer(eval_app, name="eval")
app.add_typer(mcp_app, name="mcp")


@app.callback()
def main() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    if s.phoenix_endpoint:
        # Every command (ask, and eval runs once they land) traces into Phoenix when
        # PHOENIX_ENDPOINT is set.
        import atexit

        from ledgerlens.tracing import setup_tracing, shutdown_tracing

        if setup_tracing():
            atexit.register(shutdown_tracing)  # bounded flush of pending spans at exit


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


ingest_app = typer.Typer(help="Download, parse and chunk filings.")
app.add_typer(ingest_app, name="ingest")


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


@ingest_app.command("run")
def ingest_run(
    tickers: str | None = typer.Option(
        None, help="Comma-separated tickers from the universe (default: all)."
    ),
    financebench: bool = typer.Option(
        True, "--financebench/--no-financebench", help="Also ingest FinanceBench PDFs."
    ),
    strategies: str = typer.Option("fixed,section", help="Chunk strategies to build."),
    workers: int | None = typer.Option(None, help="Parse worker processes."),
    refresh_index: bool = typer.Option(
        False,
        help="Re-download EDGAR submissions and companyfacts and update the shared "
        "snapshot in DATA_DIR/raw/sec_api (rebuild the eval datasets afterwards).",
    ),
    retry_failed: bool = typer.Option(False, help="Retry filings that failed permanently."),
) -> None:
    """Select, download, parse and chunk filings, then write the quality report.

    Exits with code 1 when any filing failed and 2 when another run holds the lock.
    """
    from ledgerlens.ingest.pipeline import DOCS_REPORT, run

    ticker_list = _csv(tickers.upper() if tickers else None)
    full_run = ticker_list is None and financebench
    try:
        summary = run(
            get_settings().data_dir,
            tickers=ticker_list,
            include_financebench=financebench,
            strategies=_csv(strategies),
            workers=workers,
            refresh_index=refresh_index,
            retry_failed=retry_failed,
            docs_report=DOCS_REPORT if full_run else None,
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"filings={len(summary.filings)} downloaded={summary.downloaded} "
        f"parsed={summary.parsed} chunked={summary.chunked} skipped={summary.skipped} "
        f"failed={len(summary.failed)} requests={summary.requests}"
    )
    for fid, err in sorted(summary.failed.items()):
        typer.echo(f"  failed {fid}: {err}", err=True)
    if summary.failed:
        raise typer.Exit(code=1)


datasets_app = typer.Typer(help="Build the eval datasets committed under eval/datasets/.")
app.add_typer(datasets_app, name="datasets")


@datasets_app.command("build")
def datasets_build(
    only: list[str] | None = typer.Option(
        None,
        "--only",
        help="Dataset to rebuild (repeatable): financebench, xbrl, unanswerable, smoke. "
        "Rebuilding xbrl also rebuilds smoke.",
    ),
) -> None:
    """Regenerate dataset JSONL files deterministically from cached sources."""
    from ledgerlens.eval.datasets.build import build_datasets

    for name, rows in build_datasets(only).items():
        typer.echo(f"{name}: {rows} rows")
