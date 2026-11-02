from ledgerlens.ingest.quality import (
    check,
    empty_sections,
    missing_items,
    outside_item_share,
    packed_cells,
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
    assert not list(tmp_path.glob("*.part"))


def test_10q_needs_part_i_items_and_flags_content_outside_items(html_ref):
    q = html_ref.model_copy(update={"form": "10-Q", "fiscal_period": "Q2"})
    blocks = [
        Block(kind="heading", text="Funding", section="Cover > Funding"),
        Block(kind="text", text="a", section="Cover > Funding"),
        Block(kind="text", text="b", section="Cover > Funding"),
        Block(kind="heading", text="Part II", section="Part II"),
        Block(kind="heading", text="Item 1. Legal Proceedings", section="Part II > Item 1. Legal"),
        Block(kind="text", text="c", section="Part II > Item 1. Legal"),
        Block(kind="heading", text="Item 2. Sales", section="Part II > Item 2. Sales"),
        Block(kind="text", text="d", section="Part II > Item 2. Sales"),
    ]
    parsed = ParsedFiling(filing=q, blocks=blocks)
    assert missing_items(parsed) == ["Part I Item 1", "Part I Item 2"]
    assert outside_item_share(parsed) == 0.5
    report = check([parsed], {})
    assert report["filings"][0]["missing_items"] == ["Part I Item 1", "Part I Item 2"]
    blocks[1:3] = [Block(kind="text", text="x", section="Cover > Funding")] * 3
    report = check([ParsedFiling(filing=q, blocks=blocks)], {})
    assert report["totals"]["filings_mostly_outside_items"] == [
        {"filing_id": q.filing_id, "share": 0.6}
    ]


def test_multi_number_cells_are_counted():
    md = "| | 2018 2017 |\n|---|---|\n| Asia | 1,624 1,672 |\n| Europe | 1,751 |"
    assert packed_cells(md) == 1
