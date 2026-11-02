"""`ledgerlens` command line entry point."""

from __future__ import annotations

import typer

from ledgerlens import __version__
from ledgerlens.config import get_settings
from ledgerlens.logging import configure_logging

app = typer.Typer(help="LedgerLens: cited answers from SEC filings, with an eval lab.")


@app.callback()
def main() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
