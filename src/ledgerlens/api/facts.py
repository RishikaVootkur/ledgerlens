"""Chunk-shaped views of XBRL facts, so the UI's source panel can show a cited fact.

The agent cites XBRL facts with ids like ``xbrl:KO:NetIncomeLoss:2025-01-01_2025-12-31:USD``
(see `ledgerlens.agent.xbrl.facts_for_concept`). The id is parsed, the fact is looked up in
SEC companyfacts, and the result is described as a short Markdown table in a `Chunk`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, get_args

from ledgerlens.agent.xbrl import Fact, _companyfacts, facts_for_concept
from ledgerlens.logging import get_logger
from ledgerlens.schemas import Chunk, FiscalPeriod, FormType
from ledgerlens.universe import BY_TICKER

log = get_logger(__name__)

_CONCEPT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,200}$")
_PERIOD = re.compile(r"^(?:(\d{4}-\d{2}-\d{2})_)?(\d{4}-\d{2}-\d{2})$")
_UNIT = re.compile(r"^[A-Za-z0-9_/.\-]{1,40}$")
_FORMS: tuple[str, ...] = get_args(FormType)
_PERIODS: tuple[str, ...] = get_args(FiscalPeriod)


@dataclass(frozen=True)
class FactId:
    raw: str
    ticker: str
    concept: str
    start: date | None
    end: date
    unit: str


def parse_fact_id(fact_id: str) -> FactId | None:
    """Parse ``xbrl:{ticker}:{concept}:{start_end | end}:{unit}``; None if malformed."""
    parts = fact_id.split(":", 4)
    if len(parts) != 5 or parts[0] != "xbrl":
        return None
    _, ticker, concept, period, unit = parts
    match = _PERIOD.match(period)
    if (
        ticker not in BY_TICKER
        or not _CONCEPT.match(concept)
        or not _UNIT.match(unit)
        or match is None
    ):
        return None
    try:
        start = date.fromisoformat(match.group(1)) if match.group(1) else None
        end = date.fromisoformat(match.group(2))
    except ValueError:
        return None
    return FactId(fact_id, ticker, concept, start, end, unit)


def find_fact(fid: FactId, data: dict[str, Any]) -> Fact | None:
    """Look the fact up in companyfacts (any taxonomy that defines the concept)."""
    for taxonomy, concepts in data.get("facts", {}).items():
        if fid.concept not in concepts:
            continue
        for fact in facts_for_concept(fid.ticker, f"{taxonomy}:{fid.concept}", data):
            if fact.fact_id == fid.raw:
                return fact
    return None


def _period_text(start: date | None, end: date) -> str:
    return f"{start.isoformat()} to {end.isoformat()}" if start else f"as of {end.isoformat()}"


def _value_text(value: float, unit: str) -> str:
    shown = f"{value:,.0f}" if value == int(value) else f"{value:,.4f}".rstrip("0")
    return f"{shown} {unit}"


def fact_chunk(fid: FactId) -> Chunk | None:
    """A Chunk describing the fact, or None when companyfacts has no such fact.

    If companyfacts cannot be read (offline, no cached file), the chunk is built from the
    id alone and says the value could not be loaded, so the citation still renders.
    """
    company = BY_TICKER[fid.ticker]
    try:
        data = _companyfacts(fid.ticker)
    except Exception as exc:
        log.warning("api.fact_lookup_failed", fact_id=fid.raw, error=type(exc).__name__)
        text = (
            f"**{fid.concept}** for {company.name}\n\n"
            "| Field | Value |\n| --- | --- |\n"
            f"| Period | {_period_text(fid.start, fid.end)} |\n"
            f"| Unit | {fid.unit} |\n\n"
            "The reported value could not be loaded from SEC companyfacts right now."
        )
        return Chunk(
            chunk_id=fid.raw,
            filing_id="",
            ticker=fid.ticker,
            company=company.name,
            form="OTHER",
            fiscal_year=fid.end.year,
            fiscal_period="FY",
            strategy="section",
            section="XBRL companyfacts",
            text=text,
        )
    fact = find_fact(fid, data)
    if fact is None:
        return None
    reported = f"{fact.form} {fact.fiscal_period or ''} {fact.fiscal_year or ''}".strip()
    text = (
        f"**{fact.label}** (`{fact.concept}`) for {company.name}\n\n"
        "| Field | Value |\n| --- | --- |\n"
        f"| Value | {_value_text(fact.value, fact.unit)} |\n"
        f"| Period | {_period_text(fact.start, fact.end)} ({fact.duration}) |\n"
        f"| Reported in | {reported}, filed {fact.filed.isoformat()} |\n"
        f"| Accession | {fact.accession} |\n\n"
        "Source: SEC XBRL companyfacts API (exact tagged value)."
    )
    return Chunk(
        chunk_id=fid.raw,
        filing_id=fact.filing_id or "",
        ticker=fact.ticker,
        company=company.name,
        form=fact.form if fact.form in _FORMS else "OTHER",  # type: ignore[arg-type]
        fiscal_year=fact.fiscal_year or fact.end.year,
        fiscal_period=fact.fiscal_period if fact.fiscal_period in _PERIODS else "FY",  # type: ignore[arg-type]
        strategy="section",
        section=f"XBRL companyfacts > {fact.label}",
        text=text,
        has_table=True,
    )
