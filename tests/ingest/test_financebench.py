import json

from ledgerlens.ingest.financebench import (
    UNKNOWN_CIK,
    dataset_path,
    filing_for_row,
    financebench_filings,
    fiscal_period_for,
)


def _row(doc_name: str, company: str, doc_type: str, period: int) -> dict[str, object]:
    return {
        "doc_name": doc_name,
        "company": company,
        "doc_type": doc_type,
        "doc_period": period,
        "doc_link": f"https://example.invalid/{doc_name}.pdf",
    }


def test_filing_for_known_company():
    ref = filing_for_row(_row("3M_2023Q2_10Q", "3M", "10q", 2023))
    assert ref.filing_id == "FB-3M_2023Q2_10Q"
    assert (ref.ticker, ref.cik, ref.form) == ("MMM", "0000066740", "10-Q")
    assert (ref.fiscal_year, ref.fiscal_period) == (2023, "Q2")
    assert ref.source_format == "pdf"
    assert ref.doc_name == "3M_2023Q2_10Q"
    assert ref.source_url.endswith("/pdfs/3M_2023Q2_10Q.pdf")


def test_form_mapping_and_unknown_company():
    ref = filing_for_row(_row("ACME_2022_8K_dated-2022-07-01", "Acme Widgets", "8k", 2022))
    assert ref.form == "8-K"
    assert ref.ticker == "ACMEWIDGET"
    assert ref.cik == UNKNOWN_CIK
    assert filing_for_row(_row("X_2023Q4_EARNINGS", "Pfizer", "Earnings", 2023)).form == "EARNINGS"


def test_fiscal_period_from_doc_name():
    assert fiscal_period_for("AMCOR_2023Q4_EARNINGS") == "Q4"
    assert fiscal_period_for("BOEING_2022_10K") == "FY"


def test_filings_are_deduplicated_by_doc_name(tmp_path):
    path = dataset_path(tmp_path)
    path.parent.mkdir(parents=True)
    rows = [
        _row("3M_2018_10K", "3M", "10k", 2018),
        _row("3M_2018_10K", "3M", "10k", 2018),
        _row("PEPSICO_2022_10K", "PepsiCo", "10k", 2022),
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    refs, links = financebench_filings(tmp_path)
    assert [r.filing_id for r in refs] == ["FB-3M_2018_10K", "FB-PEPSICO_2022_10K"]
    assert links["3M_2018_10K"] == "https://example.invalid/3M_2018_10K.pdf"
