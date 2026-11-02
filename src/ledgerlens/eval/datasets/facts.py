"""Corpus filings and XBRL fact selection from SEC submissions and companyfacts JSON.

Corpus rule (shared with ingestion): per company, the latest 10-K and the two most recent
10-Qs by filing date in `filings.recent`. Fiscal year and period come from the `fy`/`fp`
of companyfacts facts whose `accn` is the filing's accession number.

In companyfacts, `fy`/`fp` describe the *filing*, not the fact's own period: a 10-K
carries prior-year comparatives with the same `fy`. So facts are matched on `accn` plus
their own `start`/`end` dates, never on `fy` or `frame`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

from ledgerlens.schemas import make_filing_id

PeriodKind = Literal["year", "quarter", "instant"]

YEAR_DAYS = (350, 380)
# 12-week and 16-week retail quarters (Costco) as well as 13-week quarters; excludes
# year-to-date durations (about 6 or 9 months) that 10-Qs also report.
QUARTER_DAYS = (75, 125)
END_SLACK_DAYS = 7


@dataclass(frozen=True)
class CorpusFiling:
    ticker: str
    form: str
    accession: str
    filed: date
    period_end: date
    fiscal_year: int | None
    fiscal_period: str | None

    @property
    def filing_id(self) -> str | None:
        if self.fiscal_year is None or self.fiscal_period is None:
            return None
        return make_filing_id(self.ticker, self.form, self.fiscal_year, self.fiscal_period)


@dataclass(frozen=True)
class Fact:
    concept: str
    unit: str
    value: float
    start: date | None
    end: date
    accession: str

    @property
    def days(self) -> int | None:
        return None if self.start is None else (self.end - self.start).days


def fiscal_labels(companyfacts: dict[str, Any], accession: str) -> tuple[int | None, str | None]:
    """Most common (fy, fp) over all facts filed with `accession`; (None, None) if none."""
    counts: Counter[tuple[int, str]] = Counter()
    for concepts in companyfacts.get("facts", {}).values():
        for cdata in concepts.values():
            for facts in cdata.get("units", {}).values():
                for f in facts:
                    if f.get("accn") == accession and f.get("fy") and f.get("fp"):
                        counts[(int(f["fy"]), str(f["fp"]))] += 1
    if not counts:
        return None, None
    (fy, fp), _ = counts.most_common(1)[0]
    return fy, fp


def corpus_filings(
    ticker: str, submissions: dict[str, Any], companyfacts: dict[str, Any]
) -> list[CorpusFiling]:
    """The latest 10-K and two latest 10-Qs, newest first within each form."""
    recent = submissions["filings"]["recent"]
    rows = list(
        zip(
            recent["form"],
            recent["filingDate"],
            recent["accessionNumber"],
            recent["reportDate"],
            strict=True,
        )
    )
    chosen: list[tuple[str, str, str, str]] = []
    for form, keep in (("10-K", 1), ("10-Q", 2)):
        same = sorted((r for r in rows if r[0] == form), key=lambda r: r[1], reverse=True)
        chosen.extend(same[:keep])
    out = []
    for form, filed, accn, report in chosen:
        fy, fp = fiscal_labels(companyfacts, accn)
        out.append(
            CorpusFiling(
                ticker=ticker,
                form=form,
                accession=accn,
                filed=date.fromisoformat(filed),
                period_end=date.fromisoformat(report),
                fiscal_year=fy,
                fiscal_period=fp,
            )
        )
    return out


def facts_for(companyfacts: dict[str, Any], concept: str, unit: str, accession: str) -> list[Fact]:
    """All non-dimensional us-gaap facts for `concept` in `unit` filed with `accession`."""
    cdata = companyfacts.get("facts", {}).get("us-gaap", {}).get(concept)
    if not cdata:
        return []
    out = []
    for f in cdata.get("units", {}).get(unit, []):
        if f.get("accn") != accession:
            continue
        out.append(
            Fact(
                concept=concept,
                unit=unit,
                value=float(f["val"]),
                start=date.fromisoformat(f["start"]) if f.get("start") else None,
                end=date.fromisoformat(f["end"]),
                accession=accession,
            )
        )
    return out


def _period_ok(fact: Fact, kind: PeriodKind) -> bool:
    if kind == "instant":
        return fact.start is None
    if fact.days is None:
        return False
    lo, hi = YEAR_DAYS if kind == "year" else QUARTER_DAYS
    return lo <= fact.days <= hi


def match_period(facts: list[Fact], kind: PeriodKind, end: date) -> Fact | None:
    """The fact of the right period kind whose end is closest to `end` (within a week).

    Returns None when nothing matches or when matching facts disagree on the value,
    because an ambiguous gold answer is worse than no question.
    """
    cands = [f for f in facts if _period_ok(f, kind) and abs((f.end - end).days) <= END_SLACK_DAYS]
    if not cands:
        return None
    best_gap = min(abs((f.end - end).days) for f in cands)
    best = [f for f in cands if abs((f.end - end).days) == best_gap]
    if len({f.value for f in best}) != 1:
        return None
    return best[0]


def select_fact(
    companyfacts: dict[str, Any],
    concepts: tuple[str, ...],
    unit: str,
    accession: str,
    kind: PeriodKind,
    end: date,
) -> Fact | None:
    """First concept in the fallback chain with an unambiguous fact for the period."""
    for concept in concepts:
        fact = match_period(facts_for(companyfacts, concept, unit, accession), kind, end)
        if fact is not None:
            return fact
    return None


def prior_year_end(end: date) -> date:
    """Same calendar day one year earlier (52/53-week years land within the slack)."""
    try:
        return end.replace(year=end.year - 1)
    except ValueError:  # 29 February
        return end - timedelta(days=366)
