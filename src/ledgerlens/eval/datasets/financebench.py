"""FinanceBench open sample (150 questions) mapped to `Question`.

Source: `PatronusAI/financebench` on Hugging Face, pinned to a revision, CC BY-NC 4.0.
Islam et al., 2023, "FinanceBench: A New Benchmark for Financial Question Answering".

Page numbers: FinanceBench's `evidence_page_num` is 0-based. Checked by reading the PDFs
with pymupdf: for 3M_2018_10K (ids 03029, 04672), AMD_2022_10K (00222) and
AMERICANEXPRESS_2022_10K (01351) the evidence text matches PDF page index
`evidence_page_num` exactly (token overlap 1.00) and the preceding index poorly
(0.10 to 0.31), and six labels are 0. Evidence pages here are therefore `+1` (1-based).

Numeric golds: `answer_value` is set only when the whole answer is one number. Scale and
unit come from the answer ($, %) and the question ("in USD millions", "per share").
`abs_tolerance` is half a unit in the last decimal place the gold is written to, because
FinanceBench golds are rounded ("round to two decimal places").
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from ledgerlens.config import get_settings
from ledgerlens.eval.datasets.common import assign_splits, datasets_dir, write_jsonl
from ledgerlens.logging import get_logger
from ledgerlens.schemas import AnswerUnit, Evidence, FiscalPeriod, FormType, Question

log = get_logger(__name__)

REPO_ID = "PatronusAI/financebench"
FILENAME = "financebench_merged.jsonl"
REVISION = "e04404e3a97f69f79c14d42f24981a1c9c3bcd18"
PAGE_OFFSET = 1  # evidence_page_num is 0-based; Evidence.page is 1-based

FORMS: dict[str, FormType] = {"10k": "10-K", "10q": "10-Q", "8k": "8-K", "earnings": "EARNINGS"}

# Tickers for the FinanceBench companies, so answers can be filtered by company.
TICKERS = {
    "3M": "MMM",
    "AES Corporation": "AES",
    "AMD": "AMD",
    "Activision Blizzard": "ATVI",
    "Adobe": "ADBE",
    "Amazon": "AMZN",
    "Amcor": "AMCR",
    "American Express": "AXP",
    "American Water Works": "AWK",
    "Best Buy": "BBY",
    "Block": "SQ",
    "Boeing": "BA",
    "CVS Health": "CVS",
    "Coca-Cola": "KO",
    "Corning": "GLW",
    "Costco": "COST",
    "Foot Locker": "FL",
    "General Mills": "GIS",
    "JPMorgan": "JPM",
    "Johnson & Johnson": "JNJ",
    "Kraft Heinz": "KHC",
    "Lockheed Martin": "LMT",
    "MGM Resorts": "MGM",
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Nike": "NKE",
    "Paypal": "PYPL",
    "PepsiCo": "PEP",
    "Pfizer": "PFE",
    "Ulta Beauty": "ULTA",
    "Verizon": "VZ",
    "Walmart": "WMT",
}

# Golds whose unit the answer and question text do not reveal, checked by hand.
# 01319: "0" restructuring costs (an amount in USD; the question says "state 0").
UNIT_OVERRIDES: dict[str, AnswerUnit] = {"financebench_id_01319": "USD"}

_NUMBER = re.compile(
    r"^\s*(?P<neg1>-)?\s*(?P<dollar>\$)?\s*(?P<neg2>-)?\s*(?P<num>\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"\s*(?P<suffix>%|percent|million|billion|mn|bn)?\s*\.?\s*$",
    re.IGNORECASE,
)
_SCALE_WORDS = {"million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}


def _question_scale(question: str) -> float | None:
    q = question.lower()
    if re.search(r"usd\s*billions?|in\s+billions", q):
        return 1e9
    if re.search(r"usd\s*millions?|in\s+millions", q):
        return 1e6
    return None


def parse_numeric_answer(answer: str, question: str) -> tuple[float, AnswerUnit, float] | None:
    """(value in base units, unit, abs tolerance) if the answer is a single number.

    `$1577.00` to a question "in USD millions" gives (1_577_000_000, "USD", 5_000).
    `1.9%` gives (1.9, "percent", 0.05). `24.26` gives (24.26, "ratio", 0.005).
    """
    m = _NUMBER.match(answer)
    if not m:
        return None
    text = m["num"].replace(",", "")
    value = float(text)
    if m["neg1"] or m["neg2"]:
        value = -value
    decimals = len(text.split(".", 1)[1]) if "." in text else 0
    half_ulp = 0.5 * 10.0**-decimals
    suffix = (m["suffix"] or "").lower()
    q = question.lower()
    if suffix in ("%", "percent"):
        return value, "percent", half_ulp
    scale = _SCALE_WORDS.get(suffix) or _question_scale(question)
    if m["dollar"] or scale is not None:
        if "per share" in q and scale is None:
            return value, "per_share", half_ulp
        factor = scale or 1.0
        return value * factor, "USD", half_ulp * factor
    if "percent" in q or "as a %" in q or "% margin" in q:
        return value, "percent", half_ulp
    return value, "ratio", half_ulp


def fiscal_period_of(doc_name: str, form: FormType) -> FiscalPeriod | None:
    if form == "10-K":
        return "FY"
    m = re.search(r"_\d{4}(Q[1-4])_", doc_name)
    if m:
        period: FiscalPeriod = m[1]  # type: ignore[assignment]
        return period
    return None


def download(cache_dir: Path | None = None) -> list[dict[str, Any]]:
    path = hf_hub_download(
        REPO_ID,
        FILENAME,
        repo_type="dataset",
        revision=REVISION,
        cache_dir=cache_dir or get_settings().cache_dir / "huggingface",
    )
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_question(row: dict[str, Any]) -> Question:
    form = FORMS[str(row["doc_type"]).lower()]
    parsed = parse_numeric_answer(row["answer"], row["question"])
    evidence = [
        Evidence(
            doc_name=ev["doc_name"],
            page=int(ev["evidence_page_num"]) + PAGE_OFFSET,
            text=ev["evidence_text"],
        )
        for ev in row["evidence"]
    ]
    metadata: dict[str, str | int | float | bool | None] = {
        "financebench_id": row["financebench_id"],
        "fb_question_type": row["question_type"],
        "question_reasoning": row.get("question_reasoning"),
        "domain_question_num": row.get("domain_question_num"),
        "justification": row.get("justification"),
        "gics_sector": row.get("gics_sector"),
        "doc_type": row["doc_type"],
        "doc_period": row["doc_period"],
        "doc_link": row.get("doc_link"),
        "dataset_subset_label": row.get("dataset_subset_label"),
        "evidence_page_base": "evidence_page_num + 1 (source is 0-based)",
    }
    data: dict[str, Any] = {
        "qid": f"fb-{row['financebench_id'].removeprefix('financebench_id_')}",
        "dataset": "financebench",
        "split": "test",  # replaced by assign_splits
        "question": row["question"],
        "answer": row["answer"],
        "ticker": TICKERS.get(row["company"]),
        "company": row["company"],
        "fiscal_year": int(row["doc_period"]),
        "fiscal_period": fiscal_period_of(row["doc_name"], form),
        "form": form,
        "doc_name": row["doc_name"],
        "question_type": str(row["question_type"]).replace("-", "_"),
        "evidence": evidence,
        "metadata": metadata,
    }
    if parsed is not None:
        value, unit, abs_tol = parsed
        unit = UNIT_OVERRIDES.get(row["financebench_id"], unit)
        data.update(answer_value=value, answer_unit=unit, abs_tolerance=abs_tol)
    else:
        data["answer_unit"] = "text"
    return Question.model_validate(data)


def build(rows: list[dict[str, Any]] | None = None) -> list[Question]:
    questions = [to_question(r) for r in (rows if rows is not None else download())]
    qids = [q.qid for q in questions]
    assert len(qids) == len(set(qids)), "duplicate financebench qids"
    return assign_splits(questions)


def generate() -> int:
    n = write_jsonl(datasets_dir() / "financebench.jsonl", build())
    log.info("dataset_written", dataset="financebench", rows=n)
    return n
