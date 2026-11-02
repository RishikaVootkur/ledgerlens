"""Exact reported numbers from SEC's XBRL companyfacts API.

`companyfacts` lists every value a company tagged in its filings. Its `fy`/`fp` fields
describe the filing that reported a value, not the value's own period (a FY2025 10-K
also reports FY2024 comparatives), so each fact here carries its own start/end dates
and a duration label, and the model picks the period it needs.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date
from functools import lru_cache
from typing import Any

import httpx
from pydantic import BaseModel

from ledgerlens.config import get_settings
from ledgerlens.schemas import make_filing_id
from ledgerlens.universe import BY_TICKER

# Friendly metric names to candidate concepts, most specific first.
METRICS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenuesNetOfInterestExpense",
    ),
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ),
    "operating_income": ("OperatingIncomeLoss",),
    "gross_profit": ("GrossProfit",),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "rd_expense": ("ResearchAndDevelopmentExpense",),
    "sga_expense": ("SellingGeneralAndAdministrativeExpense",),
    "interest_expense": ("InterestExpense", "InterestExpenseNonoperating"),
    "income_tax": ("IncomeTaxExpenseBenefit",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "eps_basic": ("EarningsPerShareBasic",),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "stockholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "dividends_per_share": ("CommonStockDividendsPerShareDeclared",),
    "shares_outstanding": (
        "CommonStockSharesOutstanding",
        "dei:EntityCommonStockSharesOutstanding",
    ),
}

_lock = threading.Lock()
_last_request = [0.0]


class Fact(BaseModel):
    fact_id: str
    ticker: str
    concept: str
    label: str
    value: float
    unit: str
    start: date | None
    end: date
    duration: str  # FY, Q, YTD-6M, YTD-9M, instant, other
    form: str
    fiscal_year: int | None
    fiscal_period: str | None
    filing_id: str | None
    accession: str
    filed: date


def _fetch_companyfacts(cik: str) -> dict[str, Any]:
    s = get_settings()
    path = s.data_dir / "raw" / "sec_api" / f"companyfacts_CIK{cik}.json"
    if path.exists():
        data: dict[str, Any] = json.loads(path.read_text())
        return data
    with _lock:  # stay well under SEC's 10 requests per second
        wait = 0.25 - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()
    resp = httpx.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        headers={"User-Agent": s.require_sec_user_agent()},
        timeout=60,
    )
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text)
    data = resp.json()
    return data


@lru_cache(maxsize=64)
def _companyfacts(ticker: str) -> dict[str, Any]:
    company = BY_TICKER.get(ticker.upper())
    if company is None:
        raise KeyError(f"{ticker} is not in the LedgerLens universe")
    return _fetch_companyfacts(company.cik)


def _duration(start: date | None, end: date) -> str:
    if start is None:
        return "instant"
    days = (end - start).days
    if 350 <= days <= 380:
        return "FY"
    if 80 <= days <= 100:
        return "Q"
    if 170 <= days <= 195:
        return "YTD-6M"
    if 260 <= days <= 285:
        return "YTD-9M"
    return "other"


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def facts_for_concept(ticker: str, concept: str, data: dict[str, Any]) -> list[Fact]:
    taxonomy, _, name = concept.rpartition(":")
    taxonomy = taxonomy or "us-gaap"
    node = data.get("facts", {}).get(taxonomy, {}).get(name)
    if not node:
        return []
    out: dict[tuple[object, ...], Fact] = {}
    for unit, rows in node.get("units", {}).items():
        for row in rows:
            if row.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                continue
            start, end = _parse_date(row.get("start")), _parse_date(row["end"])
            assert end is not None
            form = row["form"].replace("/A", "")
            fy, fp = row.get("fy"), row.get("fp")
            filing_id = make_filing_id(ticker, form, fy, fp) if fy and fp else None
            duration = _duration(start, end)
            fact = Fact(
                fact_id=f"xbrl:{ticker}:{name}:{end.isoformat()}:{duration}",
                ticker=ticker,
                concept=f"{taxonomy}:{name}",
                label=node.get("label") or name,
                value=float(row["val"]),
                unit=unit,
                start=start,
                end=end,
                duration=duration,
                form=form,
                fiscal_year=fy,
                fiscal_period=fp,
                filing_id=filing_id,
                accession=row["accn"],
                filed=date.fromisoformat(row["filed"]),
            )
            # Same value can be re-reported in later filings; keep the earliest report
            # (the filing where it was the current period) for each period and unit.
            key = (unit, start, end)
            if key not in out or fact.filed < out[key].filed:
                out[key] = fact
    return list(out.values())


def get_company_facts(
    ticker: str,
    metric: str,
    fiscal_year: int | None = None,
    duration: str | None = None,
    limit: int = 8,
) -> list[Fact]:
    """Facts for a friendly metric name (see METRICS) or an explicit concept like
    ``us-gaap:Revenues``. Each value is attributed to the filing that first reported it, so
    `fiscal_year` keeps values from that fiscal year (not later comparatives).
    Newest period first."""
    ticker = ticker.upper()
    data = _companyfacts(ticker)
    concepts = METRICS.get(metric.lower().strip(), (metric,))
    # Companies switch concepts over time (Apple moved from Revenues to
    # RevenueFromContractWithCustomer...), so merge every candidate. For a period reported
    # under several concepts, the earlier entry in METRICS wins.
    merged: dict[tuple[object, ...], Fact] = {}
    for concept in concepts:
        for fact in facts_for_concept(ticker, concept, data):
            merged.setdefault((fact.unit, fact.start, fact.end), fact)
    facts = list(merged.values())
    if fiscal_year is not None:
        facts = [f for f in facts if f.fiscal_year == fiscal_year]
    if duration is not None:
        facts = [f for f in facts if f.duration == duration]
    facts.sort(key=lambda f: (f.end, f.duration == "FY"), reverse=True)
    return facts[:limit]
