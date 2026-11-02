"""FinanceBench open sample: the documents its 150 questions need.

The dataset (`PatronusAI/financebench`, one JSONL file) is copied once into
`DATA_DIR/raw/financebench/` so reruns stay offline. Each distinct `doc_name`
becomes one FilingRef with `filing_id = FB-{doc_name}`; the PDF comes from the
FinanceBench GitHub repo, falling back to the row's `doc_link`.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from ledgerlens.logging import get_logger
from ledgerlens.schemas import FilingRef, FiscalPeriod, FormType

log = get_logger(__name__)

HF_REPO = "PatronusAI/financebench"
HF_FILE = "financebench_merged.jsonl"
PDF_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf"
UNKNOWN_CIK = "0000000000"

FORM_MAP: dict[str, FormType] = {
    "10k": "10-K",
    "10q": "10-Q",
    "8k": "8-K",
    "earnings": "EARNINGS",
}

# FinanceBench company name -> (ticker, CIK). Checked against SEC company_tickers.json;
# ATVI and FL are delisted (acquired) and were filled from their EDGAR filer pages.
COMPANIES: dict[str, tuple[str, str]] = {
    "3M": ("MMM", "0000066740"),
    "AES Corporation": ("AES", "0000874761"),
    "AMD": ("AMD", "0000002488"),
    "Activision Blizzard": ("ATVI", "0000718877"),
    "Adobe": ("ADBE", "0000796343"),
    "Amazon": ("AMZN", "0001018724"),
    "Amcor": ("AMCR", "0001748790"),
    "American Express": ("AXP", "0000004962"),
    "American Water Works": ("AWK", "0001410636"),
    "Best Buy": ("BBY", "0000764478"),
    "Block": ("XYZ", "0001512673"),
    "Boeing": ("BA", "0000012927"),
    "CVS Health": ("CVS", "0000064803"),
    "Coca-Cola": ("KO", "0000021344"),
    "Corning": ("GLW", "0000024741"),
    "Costco": ("COST", "0000909832"),
    "Foot Locker": ("FL", "0000850209"),
    "General Mills": ("GIS", "0000040704"),
    "JPMorgan": ("JPM", "0000019617"),
    "Johnson & Johnson": ("JNJ", "0000200406"),
    "Kraft Heinz": ("KHC", "0001637459"),
    "Lockheed Martin": ("LMT", "0000936468"),
    "MGM Resorts": ("MGM", "0000789570"),
    "Microsoft": ("MSFT", "0000789019"),
    "Netflix": ("NFLX", "0001065280"),
    "Nike": ("NKE", "0000320187"),
    "Paypal": ("PYPL", "0001633917"),
    "PepsiCo": ("PEP", "0000077476"),
    "Pfizer": ("PFE", "0000078003"),
    "Ulta Beauty": ("ULTA", "0001403568"),
    "Verizon": ("VZ", "0000732712"),
    "Walmart": ("WMT", "0000104169"),
}

_QUARTER = re.compile(r"_(\d{4})(Q[1-4])_")


def slug_ticker(company: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", company.upper())[:10] or "UNKNOWN"


def dataset_path(raw_dir: Path) -> Path:
    return raw_dir / "financebench" / HF_FILE


def load_rows(raw_dir: Path) -> list[dict[str, Any]]:
    """Read the FinanceBench JSONL, downloading it from Hugging Face on first use."""
    path = dataset_path(raw_dir)
    if not path.exists():
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, path)
        log.info("financebench_downloaded", rows_file=str(path))
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fiscal_period_for(doc_name: str) -> FiscalPeriod:
    """`3M_2023Q2_10Q` -> Q2. Annual reports, 8-Ks and undated docs are FY."""
    m = _QUARTER.search(doc_name + "_")
    return m.group(2) if m else "FY"  # type: ignore[return-value]


def filing_for_row(row: dict[str, Any]) -> FilingRef:
    doc_name = str(row["doc_name"])
    company = str(row["company"])
    ticker, cik = COMPANIES.get(company, (slug_ticker(company), UNKNOWN_CIK))
    form = FORM_MAP.get(str(row["doc_type"]).lower(), "OTHER")
    return FilingRef(
        filing_id=f"FB-{doc_name}",
        cik=cik,
        ticker=ticker,
        company=company,
        form=form,
        fiscal_year=int(row["doc_period"]),
        fiscal_period=fiscal_period_for(doc_name),
        source_url=PDF_URL.format(doc_name=doc_name),
        source_format="pdf",
        doc_name=doc_name,
    )


def financebench_filings(raw_dir: Path) -> tuple[list[FilingRef], dict[str, str]]:
    """One FilingRef per distinct doc_name, plus doc_name -> fallback `doc_link`."""
    refs: dict[str, FilingRef] = {}
    links: dict[str, str] = {}
    for row in load_rows(raw_dir):
        name = str(row["doc_name"])
        if name not in refs:
            refs[name] = filing_for_row(row)
            if row.get("doc_link"):
                links[name] = str(row["doc_link"])
    unknown = sorted({r.company for r in refs.values() if r.cik == UNKNOWN_CIK})
    if unknown:
        log.warning("financebench_unknown_companies", companies=unknown)
    return [refs[k] for k in sorted(refs)], links
