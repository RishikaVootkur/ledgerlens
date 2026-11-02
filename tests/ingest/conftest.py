from __future__ import annotations

from pathlib import Path

import pytest

from ledgerlens.schemas import FilingRef

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def html_ref() -> FilingRef:
    return FilingRef(
        filing_id="SMPL-10K-2025FY",
        cik="0000000001",
        ticker="SMPL",
        company="Sample Corp",
        form="10-K",
        fiscal_year=2025,
        fiscal_period="FY",
        source_url="https://www.sec.gov/Archives/edgar/data/1/000000000125000001/sample.htm",
        source_format="html",
    )


@pytest.fixture
def pdf_ref() -> FilingRef:
    return FilingRef(
        filing_id="FB-SAMPLE_2025_10K",
        cik="0000000000",
        ticker="SMPL",
        company="Sample Corp",
        form="10-K",
        fiscal_year=2025,
        fiscal_period="FY",
        source_url="https://example.invalid/SAMPLE_2025_10K.pdf",
        source_format="pdf",
        doc_name="SAMPLE_2025_10K",
    )
