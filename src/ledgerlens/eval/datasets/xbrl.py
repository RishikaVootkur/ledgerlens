"""XBRL golden set: questions whose gold answers come straight from the filer's XBRL.

Every fact used is non-dimensional, in us-gaap, and was filed with one of the corpus
filings (see `facts.corpus_filings`), so the answer appears in an ingested document.
Three question types:

- `single_fact`: one reported value (revenue, net income, assets, EPS, ...).
- `yoy_change`: current vs prior fiscal year, both values from the same 10-K, as an
  absolute and as a percent change.
- `multi_step`: two facts plus arithmetic (margins, current ratio, liabilities to
  equity, change between the two 10-Q quarters).

Facts use rel_tolerance 0.01; derived values use 0.02, plus a small absolute tolerance
for percents and ratios near zero. Inputs and the formula are kept in `metadata`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ledgerlens.eval.datasets import sec
from ledgerlens.eval.datasets.common import (
    assign_splits,
    datasets_dir,
    fmt_date,
    fmt_usd,
    stable_hash,
    write_jsonl,
)
from ledgerlens.eval.datasets.facts import (
    CorpusFiling,
    Fact,
    PeriodKind,
    corpus_filings,
    prior_year_end,
    select_fact,
)
from ledgerlens.logging import get_logger
from ledgerlens.schemas import AnswerUnit, Evidence, Question
from ledgerlens.universe import SLICE_TICKERS, UNIVERSE, Company

log = get_logger(__name__)

FACT_TOL = 0.01
DERIVED_TOL = 0.02
PERCENT_ABS_TOL = 0.05  # percentage points
RATIO_ABS_TOL = 0.005

# Per-company caps keep the eval affordable on free LLM tiers. The slice companies are
# uncapped so the smoke set can be drawn from their dev-split questions.
CAPS = {"single_fact": 4, "yoy_change": 2, "multi_step": 3}

SHORT_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "INTC": "Intel",
    "ADBE": "Adobe",
    "AMZN": "Amazon",
    "NKE": "Nike",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "NFLX": "Netflix",
    "VZ": "Verizon",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "AXP": "American Express",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer",
    "UNH": "UnitedHealth Group",
    "XOM": "ExxonMobil",
    "CVX": "Chevron",
    "WMT": "Walmart",
    "COST": "Costco",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "BA": "Boeing",
    "LMT": "Lockheed Martin",
    "CAT": "Caterpillar",
    "MMM": "3M",
    "NEE": "NextEra Energy",
    "LIN": "Linde",
    "AMT": "American Tower",
}


@dataclass(frozen=True)
class Metric:
    key: str
    concepts: tuple[str, ...]
    label: str
    flow: bool = True  # duration (income statement) vs instant (balance sheet)
    unit: str = "USD"
    plural: bool = False
    labels: tuple[tuple[str, str], ...] = ()  # concept-specific label overrides

    def label_for(self, concept: str) -> str:
        return dict(self.labels).get(concept, self.label)

    @property
    def answer_unit(self) -> AnswerUnit:
        return "per_share" if self.unit == "USD/shares" else "USD"


REVENUE_CHAIN = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
BANK_REVENUE_CHAIN = ("RevenuesNetOfInterestExpense", "Revenues")
# Oil majors tag "Total revenues and other income" as us-gaap:Revenues.
REVENUE_LABEL_OVERRIDES = {
    "XOM": "total revenues and other income",
    "CVX": "total revenues and other income",
}

NET_INCOME = Metric(
    "net_income",
    ("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"),
    "net income",
    labels=(
        (
            "NetIncomeLossAvailableToCommonStockholdersBasic",
            "net income attributable to common stockholders",
        ),
    ),
)
OPERATING_INCOME = Metric("operating_income", ("OperatingIncomeLoss",), "operating income")
TOTAL_ASSETS = Metric("total_assets", ("Assets",), "total assets", flow=False, plural=True)
CASH = Metric(
    "cash",
    ("CashAndCashEquivalentsAtCarryingValue", "CashAndDueFromBanks"),
    "cash and cash equivalents",
    flow=False,
    plural=True,
    labels=(("CashAndDueFromBanks", "cash and due from banks"),),
)
EPS_DILUTED = Metric(
    "eps_diluted", ("EarningsPerShareDiluted",), "diluted earnings per share", unit="USD/shares"
)
RD_EXPENSE = Metric(
    "rd_expense",
    (
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ),
    "research and development expense",
)
NET_INTEREST_INCOME = Metric(
    "net_interest_income", ("InterestIncomeExpenseNet",), "net interest income"
)
NONINTEREST_INCOME = Metric("noninterest_income", ("NoninterestIncome",), "noninterest revenue")
GROSS_PROFIT = Metric("gross_profit", ("GrossProfit",), "gross profit")
CURRENT_ASSETS = Metric("current_assets", ("AssetsCurrent",), "total current assets", flow=False)
CURRENT_LIABILITIES = Metric(
    "current_liabilities", ("LiabilitiesCurrent",), "total current liabilities", flow=False
)
LIABILITIES = Metric("liabilities", ("Liabilities",), "total liabilities", flow=False)
EQUITY = Metric("equity", ("StockholdersEquity",), "total stockholders' equity", flow=False)


def revenue_metric(company: Company) -> Metric:
    if company.sector == "Financials":
        return Metric("revenue", BANK_REVENUE_CHAIN, "total net revenue")
    return Metric(
        "revenue", REVENUE_CHAIN, REVENUE_LABEL_OVERRIDES.get(company.ticker, "total revenue")
    )


def single_fact_metrics(company: Company, form: str) -> list[Metric]:
    rev = revenue_metric(company)
    if form == "10-K":
        metrics = [rev, NET_INCOME, OPERATING_INCOME, TOTAL_ASSETS, CASH, EPS_DILUTED, RD_EXPENSE]
    else:
        metrics = [rev, NET_INCOME, OPERATING_INCOME, EPS_DILUTED, TOTAL_ASSETS, CASH]
    if company.sector == "Financials":
        metrics += [NET_INTEREST_INCOME, NONINTEREST_INCOME]
    return metrics


@dataclass(frozen=True)
class Found:
    """A selected fact plus the context needed to cite it."""

    metric: Metric
    fact: Fact
    filing: CorpusFiling

    @property
    def label(self) -> str:
        return self.metric.label_for(self.fact.concept)

    @property
    def filing_id(self) -> str:
        fid = self.filing.filing_id
        assert fid is not None
        return fid


class CompanyFacts:
    """Fact lookups for one company's corpus filings."""

    def __init__(self, company: Company, companyfacts: dict[str, Any], filings: list[CorpusFiling]):
        self.company = company
        self.cf = companyfacts
        self.filings = [f for f in filings if f.filing_id is not None]
        self.tenk = next((f for f in self.filings if f.form == "10-K"), None)
        self.tenqs = [f for f in self.filings if f.form == "10-Q"]

    def find(
        self,
        metric: Metric,
        filing: CorpusFiling,
        *,
        prior_year: bool = False,
        concepts: tuple[str, ...] | None = None,
    ) -> Found | None:
        kind: PeriodKind = (
            "instant" if not metric.flow else "year" if filing.form == "10-K" else "quarter"
        )
        end = prior_year_end(filing.period_end) if prior_year else filing.period_end
        fact = select_fact(
            self.cf, concepts or metric.concepts, metric.unit, filing.accession, kind, end
        )
        return None if fact is None else Found(metric, fact, filing)


# ---------------------------------------------------------------- phrasing


def possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def period_phrase(filing: CorpusFiling, metric: Metric, *, prior_year: bool = False) -> str:
    assert filing.fiscal_year is not None
    if filing.form == "10-K":
        fy = filing.fiscal_year - (1 if prior_year else 0)
        end = prior_year_end(filing.period_end) if prior_year else filing.period_end
        if metric.flow:
            return f"for fiscal year {fy} (the year ended {fmt_date(end)})"
        return f"as of the end of fiscal year {fy} ({fmt_date(end)})"
    quarter = f"fiscal {filing.fiscal_year} {filing.fiscal_period}"
    if metric.flow:
        return f"for the three months ended {fmt_date(filing.period_end)} ({quarter})"
    return f"as of {fmt_date(filing.period_end)} (end of {quarter})"


def unit_suffix(metric: Metric) -> str:
    return "in US dollars per share" if metric.unit == "USD/shares" else "in US dollars"


def value_text(value: float, unit: AnswerUnit) -> str:
    if unit == "USD":
        return fmt_usd(value)
    if unit == "per_share":
        return f"${value:.2f} per diluted share"
    if unit == "percent":
        return f"{value:.2f}%"
    return f"{value:.2f}"


def _base(company: Company, filing: CorpusFiling) -> dict[str, Any]:
    return {
        "dataset": "xbrl",
        "split": "test",  # replaced by assign_splits
        "ticker": company.ticker,
        "company": company.name,
        "fiscal_year": filing.fiscal_year,
        "fiscal_period": filing.fiscal_period,
        "form": filing.form,
    }


def _inputs_meta(found: list[Found]) -> dict[str, str | int | float | bool | None]:
    meta: dict[str, str | int | float | bool | None] = {}
    for i, f in enumerate(found, start=1):
        meta[f"input{i}_concept"] = f"us-gaap:{f.fact.concept}"
        meta[f"input{i}_value"] = f.fact.value
        meta[f"input{i}_unit"] = f.fact.unit
        meta[f"input{i}_start"] = f.fact.start.isoformat() if f.fact.start else None
        meta[f"input{i}_end"] = f.fact.end.isoformat()
        meta[f"input{i}_filing_id"] = f.filing_id
        meta[f"input{i}_accession"] = f.fact.accession
    return meta


def _evidence(found: list[Found]) -> list[Evidence]:
    seen: list[Evidence] = []
    for f in found:
        ev = Evidence(filing_id=f.filing_id, xbrl_concept=f"us-gaap:{f.fact.concept}")
        if ev not in seen:
            seen.append(ev)
    return seen


def _slug(filing: CorpusFiling) -> str:
    fid = filing.filing_id
    assert fid is not None
    return fid.split("-", 1)[1].replace("-", "").lower()  # 10k2025fy


# ---------------------------------------------------------------- generators


def single_fact_questions(cf: CompanyFacts) -> list[Question]:
    out = []
    name = possessive(SHORT_NAMES[cf.company.ticker])
    for filing in cf.filings:
        for metric in single_fact_metrics(cf.company, filing.form):
            found = cf.find(metric, filing)
            if found is None:
                continue
            verb = "were" if metric.plural else "was"
            question = (
                f"What {verb} {name} {found.label} {period_phrase(filing, metric)}, "
                f"{unit_suffix(metric)}?"
            )
            unit = metric.answer_unit
            out.append(
                Question.model_validate(
                    {
                        **_base(cf.company, filing),
                        "qid": f"xbrl-{cf.company.ticker.lower()}-{_slug(filing)}-{metric.key}",
                        "question": question,
                        "answer": value_text(found.fact.value, unit),
                        "answer_value": found.fact.value,
                        "answer_unit": unit,
                        "rel_tolerance": FACT_TOL,
                        "question_type": "single_fact",
                        "evidence": _evidence([found]),
                        "metadata": {
                            "metric": metric.key,
                            "formula": "reported value",
                            **_inputs_meta([found]),
                        },
                    }
                )
            )
    return out


def pct_change(new: float, old: float) -> float:
    """Percent change from `old` to `new`, e.g. 12.5 for +12.5%."""
    return (new - old) / abs(old) * 100.0


def yoy_questions(cf: CompanyFacts) -> list[Question]:
    if cf.tenk is None:
        return []
    filing = cf.tenk
    assert filing.fiscal_year is not None
    name = possessive(SHORT_NAMES[cf.company.ticker])
    out = []
    metrics = [revenue_metric(cf.company), NET_INCOME, OPERATING_INCOME, TOTAL_ASSETS]
    for metric in metrics:
        cur = cf.find(metric, filing)
        if cur is None:
            continue
        prev = cf.find(metric, filing, prior_year=True, concepts=(cur.fact.concept,))
        if prev is None or prev.fact.value == cur.fact.value:
            continue
        fy, py = filing.fiscal_year, filing.fiscal_year - 1
        span = (
            f"from fiscal year {py} to fiscal year {fy}"
            if metric.flow
            else f"from the end of fiscal year {py} to the end of fiscal year {fy}"
        )
        label = cur.label
        meta = {"metric": metric.key, **_inputs_meta([cur, prev])}
        base = {
            **_base(cf.company, filing),
            "question_type": "yoy_change",
            "evidence": _evidence([cur, prev]),
            "rel_tolerance": DERIVED_TOL,
        }
        stem = f"xbrl-{cf.company.ticker.lower()}-{_slug(filing)}-{metric.key}-yoy"
        diff = cur.fact.value - prev.fact.value
        out.append(
            Question.model_validate(
                {
                    **base,
                    "qid": f"{stem}-abs",
                    "question": (
                        f"By how much did {name} {label} change {span}, in US dollars? "
                        "Give the signed change (negative for a decrease)."
                    ),
                    "answer": value_text(diff, "USD"),
                    "answer_value": diff,
                    "answer_unit": "USD",
                    "metadata": {
                        **meta,
                        "metric": f"{metric.key}_yoy_abs",
                        "formula": "input1 - input2",
                    },
                }
            )
        )
        if prev.fact.value > 0:
            pct = round(pct_change(cur.fact.value, prev.fact.value), 4)
            out.append(
                Question.model_validate(
                    {
                        **base,
                        "qid": f"{stem}-pct",
                        "question": (
                            f"By what percentage did {name} {label} change {span}? "
                            "Give the signed percent change."
                        ),
                        "answer": value_text(pct, "percent"),
                        "answer_value": pct,
                        "answer_unit": "percent",
                        "abs_tolerance": PERCENT_ABS_TOL,
                        "metadata": {
                            **meta,
                            "metric": f"{metric.key}_yoy_pct",
                            "formula": "(input1 - input2) / input2 * 100",
                        },
                    }
                )
            )
    return out


@dataclass(frozen=True)
class Ratio:
    key: str
    numerator: Metric
    denominator: Metric | None  # None means revenue
    unit: AnswerUnit
    phrase: Callable[[str, str], str]  # (numerator label, denominator label) -> description


RATIOS = (
    Ratio(
        "net_margin",
        NET_INCOME,
        None,
        "percent",
        lambda n, d: f"net profit margin ({n} divided by {d}, as a percentage)",
    ),
    Ratio(
        "gross_margin",
        GROSS_PROFIT,
        None,
        "percent",
        lambda n, d: f"gross margin ({n} divided by {d}, as a percentage)",
    ),
    Ratio(
        "operating_margin",
        OPERATING_INCOME,
        None,
        "percent",
        lambda n, d: f"operating margin ({n} divided by {d}, as a percentage)",
    ),
    Ratio(
        "current_ratio",
        CURRENT_ASSETS,
        CURRENT_LIABILITIES,
        "ratio",
        lambda n, d: f"current ratio ({n} divided by {d})",
    ),
    Ratio(
        "liabilities_to_equity",
        LIABILITIES,
        EQUITY,
        "ratio",
        lambda n, d: f"ratio of {n} to {d}",
    ),
)


def ratio_value(numerator: float, denominator: float, unit: AnswerUnit) -> float:
    value = numerator / denominator
    return round(value * 100.0 if unit == "percent" else value, 4)


def ratio_questions(cf: CompanyFacts) -> list[Question]:
    """Margins and balance-sheet ratios from the 10-K and from the latest 10-Q."""
    name = possessive(SHORT_NAMES[cf.company.ticker])
    filings = [f for f in (cf.tenk, cf.tenqs[0] if cf.tenqs else None) if f is not None]
    out = []
    for filing in filings:
        for ratio in RATIOS:
            den_metric = ratio.denominator or revenue_metric(cf.company)
            num = cf.find(ratio.numerator, filing)
            den = cf.find(den_metric, filing)
            if num is None or den is None or den.fact.value <= 0:
                continue
            value = ratio_value(num.fact.value, den.fact.value, ratio.unit)
            desc = ratio.phrase(num.label, den.label)
            period = period_phrase(filing, ratio.numerator)
            out.append(
                Question.model_validate(
                    {
                        **_base(cf.company, filing),
                        "qid": f"xbrl-{cf.company.ticker.lower()}-{_slug(filing)}-{ratio.key}",
                        "question": f"What was {name} {desc} {period}?",
                        "answer": value_text(value, ratio.unit),
                        "answer_value": value,
                        "answer_unit": ratio.unit,
                        "rel_tolerance": DERIVED_TOL,
                        "abs_tolerance": PERCENT_ABS_TOL
                        if ratio.unit == "percent"
                        else RATIO_ABS_TOL,
                        "question_type": "multi_step",
                        "evidence": _evidence([num, den]),
                        "metadata": {
                            "metric": ratio.key,
                            "formula": "input1 / input2"
                            + (" * 100" if ratio.unit == "percent" else ""),
                            **_inputs_meta([num, den]),
                        },
                    }
                )
            )
    return out


def qoq_questions(cf: CompanyFacts) -> list[Question]:
    """Percent change between the two corpus 10-Q quarters (three-month periods)."""
    if len(cf.tenqs) < 2:
        return []
    late, early = cf.tenqs[0], cf.tenqs[1]
    name = possessive(SHORT_NAMES[cf.company.ticker])
    out = []
    for metric in (revenue_metric(cf.company), NET_INCOME):
        cur = cf.find(metric, late)
        if cur is None:
            continue
        prev = cf.find(metric, early, concepts=(cur.fact.concept,))
        if prev is None or prev.fact.value <= 0 or prev.fact.value == cur.fact.value:
            continue
        pct = round(pct_change(cur.fact.value, prev.fact.value), 4)
        q = (
            f"By what percentage did {name} quarterly {cur.label} change from the three "
            f"months ended {fmt_date(early.period_end)} (fiscal {early.fiscal_year} "
            f"{early.fiscal_period}) to the three months ended {fmt_date(late.period_end)} "
            f"(fiscal {late.fiscal_year} {late.fiscal_period})? Give the signed percent change."
        )
        out.append(
            Question.model_validate(
                {
                    **_base(cf.company, late),
                    "qid": f"xbrl-{cf.company.ticker.lower()}-{_slug(late)}-{metric.key}-qoq-pct",
                    "question": q,
                    "answer": value_text(pct, "percent"),
                    "answer_value": pct,
                    "answer_unit": "percent",
                    "rel_tolerance": DERIVED_TOL,
                    "abs_tolerance": PERCENT_ABS_TOL,
                    "question_type": "multi_step",
                    "evidence": _evidence([cur, prev]),
                    "metadata": {
                        "metric": f"{metric.key}_qoq_pct",
                        "formula": "(input1 - input2) / input2 * 100",
                        **_inputs_meta([cur, prev]),
                    },
                }
            )
        )
    return out


def cap_per_type(questions: list[Question], caps: dict[str, int]) -> list[Question]:
    """Keep at most caps[type] questions per type, chosen by SHA-256 of "cap:{qid}".

    The salt keeps the cap independent of the split hash; unsalted, capped companies
    would keep only their lowest-hash (dev-leaning) questions and skew the splits.
    """
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q)
    keep: set[str] = set()
    for qtype, qs in by_type.items():
        ordered = sorted(qs, key=lambda q: stable_hash(f"cap:{q.qid}"))
        keep.update(q.qid for q in ordered[: caps.get(qtype, len(ordered))])
    return [q for q in questions if q.qid in keep]


def company_questions(
    company: Company, submissions: dict[str, Any], facts: dict[str, Any]
) -> list[Question]:
    filings = corpus_filings(company.ticker, submissions, facts)
    for f in filings:
        if f.filing_id is None:
            log.warning(
                "filing_without_xbrl_facts",
                ticker=company.ticker,
                form=f.form,
                accession=f.accession,
            )
    cf = CompanyFacts(company, facts, filings)
    qs = single_fact_questions(cf) + yoy_questions(cf) + ratio_questions(cf) + qoq_questions(cf)
    if company.ticker in SLICE_TICKERS:
        return qs
    return cap_per_type(qs, CAPS)


def build(companies: tuple[Company, ...] = UNIVERSE) -> list[Question]:
    questions: list[Question] = []
    for company in companies:
        qs = company_questions(company, sec.submissions(company.cik), sec.companyfacts(company.cik))
        if not qs:
            log.warning("company_skipped", ticker=company.ticker, reason="no usable facts")
        questions.extend(qs)
    qids = [q.qid for q in questions]
    assert len(qids) == len(set(qids)), "duplicate xbrl qids"
    return assign_splits(questions)


def generate() -> int:
    questions = build()
    n = write_jsonl(datasets_dir() / "xbrl.jsonl", questions)
    log.info("dataset_written", dataset="xbrl", rows=n)
    return n
