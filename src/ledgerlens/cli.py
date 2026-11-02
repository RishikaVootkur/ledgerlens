"""`ledgerlens` command line entry point."""

from __future__ import annotations

import typer

from ledgerlens import __version__
from ledgerlens.agent.cli import ask
from ledgerlens.config import get_settings
from ledgerlens.logging import configure_logging
from ledgerlens.retrieval.cli import index_app

app = typer.Typer(help="LedgerLens: cited answers from SEC filings, with an eval lab.")
app.add_typer(index_app, name="index")
app.command("ask")(ask)


@app.callback()
def main() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
