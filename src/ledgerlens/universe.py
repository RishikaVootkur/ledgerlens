"""The companies LedgerLens ingests from EDGAR, spread across sectors."""

from __future__ import annotations

from typing import NamedTuple


class Company(NamedTuple):
    ticker: str
    cik: str
    name: str
    sector: str


UNIVERSE: tuple[Company, ...] = (
    Company("AAPL", "0000320193", "Apple Inc.", "Information Technology"),
    Company("MSFT", "0000789019", "Microsoft Corporation", "Information Technology"),
    Company("NVDA", "0001045810", "NVIDIA Corporation", "Information Technology"),
    Company("INTC", "0000050863", "Intel Corporation", "Information Technology"),
    Company("ADBE", "0000796343", "Adobe Inc.", "Information Technology"),
    Company("AMZN", "0001018724", "Amazon.com, Inc.", "Consumer Discretionary"),
    Company("NKE", "0000320187", "NIKE, Inc.", "Consumer Discretionary"),
    Company("GOOGL", "0001652044", "Alphabet Inc.", "Communication Services"),
    Company("META", "0001326801", "Meta Platforms, Inc.", "Communication Services"),
    Company("NFLX", "0001065280", "Netflix, Inc.", "Communication Services"),
    Company("VZ", "0000732712", "Verizon Communications Inc.", "Communication Services"),
    Company("JPM", "0000019617", "JPMorgan Chase & Co.", "Financials"),
    Company("BAC", "0000070858", "Bank of America Corporation", "Financials"),
    Company("AXP", "0000004962", "American Express Company", "Financials"),
    Company("JNJ", "0000200406", "Johnson & Johnson", "Health Care"),
    Company("PFE", "0000078003", "Pfizer Inc.", "Health Care"),
    Company("UNH", "0000731766", "UnitedHealth Group Incorporated", "Health Care"),
    Company("XOM", "0000034088", "Exxon Mobil Corporation", "Energy"),
    Company("CVX", "0000093410", "Chevron Corporation", "Energy"),
    Company("WMT", "0000104169", "Walmart Inc.", "Consumer Staples"),
    Company("COST", "0000909832", "Costco Wholesale Corporation", "Consumer Staples"),
    Company("KO", "0000021344", "The Coca-Cola Company", "Consumer Staples"),
    Company("PEP", "0000077476", "PepsiCo, Inc.", "Consumer Staples"),
    Company("BA", "0000012927", "The Boeing Company", "Industrials"),
    Company("LMT", "0000936468", "Lockheed Martin Corporation", "Industrials"),
    Company("CAT", "0000018230", "Caterpillar Inc.", "Industrials"),
    Company("MMM", "0000066740", "3M Company", "Industrials"),
    Company("NEE", "0000753308", "NextEra Energy, Inc.", "Utilities"),
    Company("LIN", "0001707925", "Linde plc", "Materials"),
    Company("AMT", "0001053507", "American Tower Corporation", "Real Estate"),
)

# The thin end-to-end slice uses these three first.
SLICE_TICKERS: tuple[str, ...] = ("AAPL", "JPM", "KO")

BY_TICKER: dict[str, Company] = {c.ticker: c for c in UNIVERSE}
