from ledgerlens.ingest.quality import (
    check,
    empty_sections,
    missing_items,
    table_problems,
    to_markdown,
    write_reports,
)
from ledgerlens.schemas import Block, Chunk, ParsedFiling


def _chunk(fid: str, seq: int, text: str, strategy: str = "section") -> Chunk:
    return Chunk(
        chunk_id=f"{fid}:{strategy}:{seq:05d}",
        filing_id=fid,
        ticker="SMPL",
        company="Sample Corp",
        form="10-K",
        fiscal_year=2025,
        fiscal_period="FY",
        strategy=strategy,  # type: ignore[arg-type]
        section="Item 7. MD&A",
        text=text,
    )


def test_table_problems():
    good = "| a | b |\n|---|---|\n| x | 1 |"
    assert table_problems(good) == []
    assert "ragged_rows" in table_problems("| a | b |\n|---|---|\n| x | 1 | 2 |")
    assert "empty_column" in table_problems("| a |  |\n|---|---|\n| x |  |")
    assert "header_only" in table_problems("| a | b |\n|---|---|")


def test_empty_sections_and_missing_items(html_ref):
    blocks = [
        Block(kind="heading", text="Part I", section="Part I"),
        Block(kind="heading", text="Item 1. Business", section="Item 1. Business"),
        Block(kind="heading", text="Item 6. [Reserved]", section="Item 6. [Reserved]"),
        Block(kind="heading", text="Item 7. MD&A", section="Item 7. MD&A"),
        Block(kind="heading", text="Overview", section="Item 7. MD&A > Overview"),
        Block(kind="heading", text="Results", section="Item 7. MD&A > Results"),
        Block(kind="text", text="Sales rose.", section="Item 7. MD&A > Results"),
    ]
    parsed = ParsedFiling(filing=html_ref, blocks=blocks)
    empty, empty_subs = empty_sections(parsed)
    assert empty == ["Item 1. Business", "Item 6. [Reserved]"]  # Part I is structural
    assert empty_subs == ["Item 7. MD&A > Overview"]
    assert missing_items(parsed) == ["Item 8"]


def test_duplicates_and_size_bounds(html_ref, tmp_path):
    long_text = " ".join(["word"] * 1000)
    header_a = "[SMPL 10-K FY2025 | Item 7. MD&A | p.1]\n"
    header_b = "[OTHR 10-K FY2025 | Item 7. MD&A | p.9]\n"
    chunks = [
        _chunk("A", 0, header_a + "Revenue rose 5% on strong demand across regions this year."),
        _chunk("B", 0, header_b + "Revenue rose 5% on strong demand across regions this year."),
        _chunk("B", 1, header_b + "REVENUE rose 5%, on strong demand across regions this year!"),
        _chunk("A", 1, header_a + "tiny"),
        _chunk("A", 2, header_a + long_text),
    ]
    parsed = ParsedFiling(filing=html_ref, blocks=[])
    report = check([parsed], {"section": chunks})
    s = report["strategies"]["section"]
    assert s["exact_duplicates"]["groups"] == 1
    assert s["exact_duplicates"]["extra_chunks"] == 1
    assert s["exact_duplicates"]["cross_filing_groups"] == 1
    assert s["near_duplicates"]["groups"] == 1
    assert s["near_duplicates"]["extra_chunks"] == 2
    assert s["under_min"] == 1
    assert s["over_max"] == 1
    md = to_markdown(report)
    assert "| section | 5 |" in md
    json_path, md_path = write_reports(report, tmp_path)
    assert json_path.exists() and md_path.read_text() == md
