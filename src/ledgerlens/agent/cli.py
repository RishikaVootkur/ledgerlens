"""`ledgerlens ask` command."""

from __future__ import annotations

import typer

from ledgerlens.retrieval.filters import SearchFilters


def ask(
    question: str,
    model: str = typer.Option("gemini-3.8-flash", help="Model alias from ledgerlens.llm.MODELS"),
    ticker: str | None = typer.Option(None, help="Restrict search to one company"),
    year: int | None = typer.Option(None, help="Restrict search to one fiscal year"),
    method: str = typer.Option("hybrid", help="bm25, dense, hybrid or hybrid_rerank"),
    no_xbrl: bool = typer.Option(False, "--no-xbrl", help="Disable the XBRL facts tool"),
) -> None:
    """Ask the agent a question and print a cited answer."""
    from ledgerlens.agent.graph import AgentConfig, run_agent
    from ledgerlens.agent.tools import RetrievalConfig
    from ledgerlens.tracing import setup_tracing, shutdown_tracing

    traced = setup_tracing()  # no-op unless PHOENIX_ENDPOINT is set

    config = AgentConfig(
        model=model,
        retrieval=RetrievalConfig(method=method),  # type: ignore[arg-type]
        xbrl_tool=not no_xbrl,
    )
    try:
        result = run_agent(
            question,
            config,
            SearchFilters(ticker=ticker, fiscal_year=year),
            on_step=lambda name, detail: typer.secho(f"  · {name} {detail}", fg="bright_black"),
        )
    finally:
        if traced:
            shutdown_tracing()  # flush spans before the process exits
    typer.echo("\n" + result.answer + "\n")
    for i, c in enumerate(result.citations, start=1):
        typer.echo(f"[{i}] {c.filing_id} | {c.section[:70]} | p.{c.page} | {c.quote[:90]}")
    u = result.usage
    typer.secho(
        f"\n{u.llm_calls} LLM calls, {u.prompt_tokens + u.completion_tokens} tokens, "
        f"{u.latency_ms / 1000:.1f}s model time, ${u.cost_usd:.4f} list-price equivalent",
        fg="bright_black",
    )
