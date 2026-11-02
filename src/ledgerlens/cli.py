"""`ledgerlens` command line entry point."""

from __future__ import annotations

import typer

from ledgerlens import __version__
from ledgerlens.agent.cli import ask
from ledgerlens.config import get_settings
from ledgerlens.eval.cli import eval_app
from ledgerlens.logging import configure_logging
from ledgerlens.retrieval.cli import index_app

app = typer.Typer(help="LedgerLens: cited answers from SEC filings, with an eval lab.")
app.add_typer(index_app, name="index")
app.command("ask")(ask)
app.add_typer(eval_app, name="eval")


@app.callback()
def main() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


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
