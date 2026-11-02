import pytest

from ledgerlens.schemas import Chunk


def make_chunk(
    cid: str, text: str, ticker="AAPL", year=2025, form="10-K", page=1, strategy="section"
):
    filing = f"{ticker}-{form.replace('-', '')}-{year}FY"
    return Chunk(
        chunk_id=f"{filing}:{strategy}:{cid}",
        filing_id=filing,
        ticker=ticker,
        company=ticker,
        form=form,
        fiscal_year=year,
        fiscal_period="FY",
        strategy=strategy,
        section="Item 7",
        page=page,
        text=text,
    )


@pytest.fixture
def corpus():
    return [
        make_chunk("00001", "Net sales increased to $391 billion driven by iPhone and Services."),
        make_chunk("00002", "Research and development expense was $31.4 billion."),
        make_chunk("00003", "Total net revenue for the bank was $158 billion.", ticker="JPM"),
        make_chunk("00004", "Coca-Cola unit case volume grew 2% in fiscal 2025.", ticker="KO"),
        make_chunk("00005", "iPhone net sales were $201 billion in fiscal 2024.", year=2024),
    ]
