"""`ledgerlens index ...` and `ledgerlens search ...` commands."""

from __future__ import annotations

import typer

from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.schemas import ChunkStrategy

index_app = typer.Typer(help="Build and query the retrieval indexes.")


@index_app.command("build")
def build(
    strategies: str = typer.Option("section,fixed", help="Comma-separated chunk strategies"),
    filings: str = typer.Option("", help="Only these filing ids (comma-separated)"),
    dense: bool = typer.Option(True, help="Embed into Qdrant (needs Ollama and Qdrant)"),
) -> None:
    """Load chunks into SQLite, rebuild BM25 and embed new chunks into Qdrant."""
    from ledgerlens.retrieval.indexer import build_index

    only = {f.strip() for f in filings.split(",") if f.strip()} or None
    for strategy in strategies.split(","):
        stats = build_index(_strategy(strategy), filing_ids=only, dense=dense)
        typer.echo(f"{strategy}: {stats}")


@index_app.command("search")
def search(
    query: str,
    k: int = 5,
    strategy: str = "section",
    method: str = "hybrid",
    ticker: str | None = None,
    year: int | None = None,
    form: str | None = None,
) -> None:
    """Run one query and print the top hits."""
    from ledgerlens.retrieval.retriever import METHODS, Retriever

    if method not in METHODS:
        raise typer.BadParameter(f"method must be one of {METHODS}")
    retriever = Retriever(_strategy(strategy), method)  # type: ignore[arg-type]
    filters = SearchFilters.model_validate({"ticker": ticker, "fiscal_year": year, "form": form})
    hits = retriever.search(query, k, filters)
    for hit in hits:
        c = hit.chunk
        typer.echo(f"{hit.rank:>2}. {hit.score:8.4f}  {c.chunk_id}  p.{c.page}  {c.section[:60]}")
        typer.echo("    " + c.text[:200].replace("\n", " "))


def _strategy(value: str) -> ChunkStrategy:
    value = value.strip()
    if value not in ("fixed", "section"):
        raise typer.BadParameter("strategy must be 'fixed' or 'section'")
    return value  # type: ignore[return-value]
