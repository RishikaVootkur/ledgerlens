from ledgerlens.ingest.parse_html import parse_html
from ledgerlens.ingest.sections import RawItem, build_blocks, match_item, match_part


def _parse(fixtures, html_ref):
    return parse_html((fixtures / "edgar_sample.htm").read_bytes(), html_ref)


def test_hidden_xbrl_header_is_dropped(fixtures, html_ref):
    parsed = _parse(fixtures, html_ref)
    assert not any("HIDDEN XBRL" in b.text for b in parsed.blocks)


def test_headings_and_section_paths(fixtures, html_ref):
    parsed = _parse(fixtures, html_ref)
    headings = [(b.text, b.section) for b in parsed.blocks if b.kind == "heading"]
    item7 = (
        "Item 7. Management’s Discussion and Analysis of Financial Condition"
        " and Results of Operations"
    )
    assert ("Item 1. Business", "Item 1. Business") in headings
    assert (item7, item7) in headings
    assert (
        "Liquidity and Capital Resources",
        f"{item7} > Liquidity and Capital Resources",
    ) in headings
    assert ("Part II", "Part II") in headings
    text = next(b for b in parsed.blocks if b.text.startswith("The company holds"))
    assert text.section == f"{item7} > Liquidity and Capital Resources"


def test_toc_table_does_not_create_headings(fixtures, html_ref):
    parsed = _parse(fixtures, html_ref)
    item1 = [b for b in parsed.blocks if b.kind == "heading" and b.text.startswith("Item 1.")]
    assert len(item1) == 1


def test_page_estimation_from_breaks(fixtures, html_ref):
    parsed = _parse(fixtures, html_ref)
    page = {b.text: b.page for b in parsed.blocks}
    assert page["Sample Corp annual report cover page."] == 1
    assert page["Item 1. Business"] == 2  # after <hr style="page-break-after">
    assert page["Part II"] == 3  # div with page-break-before
    assert page["Item 8. Financial Statements and Supplementary Data"] == 4  # break-after:page


def test_repeated_footers_are_removed(fixtures, html_ref):
    parsed = _parse(fixtures, html_ref)
    assert not any("2025 Form 10-K |" in b.text for b in parsed.blocks)


def test_split_cell_table_becomes_markdown(fixtures, html_ref):
    parsed = _parse(fixtures, html_ref)
    tables = [b for b in parsed.blocks if b.kind == "table"]
    assert len(tables) == 2  # TOC and the data table
    data = tables[-1]
    assert data.table_id == "SMPL-10K-2025FY:table:0002"
    assert data.page == 3
    assert data.text.splitlines() == [
        "|  | 2025 | 2024 |",
        "|---|---|---|",
        "| Net sales | $1,234 | $1,100 |",
        "| Other income/(expense), net | (321) | 45 |",
        "| Gross margin | 40% | 38% |",
    ]


def test_item_and_part_patterns():
    assert match_item("ITEM 1A. RISK FACTORS") == ("1A", "Risk Factors")
    assert match_item("Item 7 — MD&A") == ("7", "MD&A")
    assert match_item("Item 7.") == ("7", "")
    assert match_item("Item 1 Business") == ("1", "Business")
    assert match_item("Item 7 of this report discusses results") is None
    assert match_item("Item 7. Management’s Discussion 25") is None  # TOC line
    assert match_part("PART II — OTHER INFORMATION") == ("II", "Other Information")
    assert match_part("Part III") == ("III", "")


def test_text_toc_run_is_demoted_and_10q_paths_include_part():
    items = [
        RawItem("para", "Item 1. Financial Statements", 1, strong=True),
        RawItem("para", "Item 2. MD&A", 1, strong=True),
        RawItem("para", "Item 3. Market Risk", 1, strong=True),
        RawItem("para", "PART I", 2, strong=True),
        RawItem("para", "Item 2. MD&A", 2, strong=True),
        RawItem("para", "Revenue rose.", 2),
    ]
    blocks = build_blocks(items, "10-Q", n_pages=2)
    headings = [b.section for b in blocks if b.kind == "heading"]
    assert headings == ["Part I", "Part I > Item 2. MD&A"]
    assert blocks[-1].section == "Part I > Item 2. MD&A"
