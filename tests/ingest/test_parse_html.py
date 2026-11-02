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


def test_column_header_lines_are_not_subheadings():
    from ledgerlens.ingest.sections import is_subheading

    assert not is_subheading("(Millions) 2018 2017 2016")
    assert not is_subheading("2025 2024")
    assert is_subheading("Note 1. Significant Accounting Policies")
    assert is_subheading("2022 Employee Stock Plan")


def _para(text, page, strong=False, edge=False):
    return RawItem("para", text, page, strong=strong, edge=edge)


def test_cover_headings_nest_under_cover_and_reset_at_first_item():
    items = [
        _para("Documents Incorporated By Reference", 1, strong=True),
        _para("Portions of the proxy statement are incorporated.", 1),
        _para("Item 1. Business", 2, strong=True),
        _para("We make widgets.", 2),
    ]
    blocks = build_blocks(items, "10-K", n_pages=2)
    sections = [b.section for b in blocks]
    assert sections[:2] == ["Cover > Documents Incorporated By Reference"] * 2
    assert sections[-1] == "Item 1. Business"


def test_title_anchor_maps_10q_without_item_headings():
    items = [
        _para("Forward-looking Statements", 1, strong=True),
        _para("This report contains forward-looking statements.", 1),
        _para("Consolidated Statement of Income", 2, strong=True),
        _para("Revenue was 10.", 2),
        _para("Consolidated balance sheets analysis", 3, strong=True),
        _para("Assets rose.", 3),
        _para("Item 2. Management's Discussion and Analysis", 4, strong=True),
        _para("Sales grew.", 4),
    ]
    blocks = build_blocks(items, "10-Q", n_pages=4)
    by_text = {b.text: b.section for b in blocks}
    assert by_text["This report contains forward-looking statements."] == (
        "Cover > Forward-looking Statements"
    )
    fin = "Part I > Item 1. Financial Statements"
    assert by_text["Revenue was 10."] == f"{fin} > Consolidated Statement of Income"
    # "... analysis" is not a full-title match, so it stays a sub-heading of Item 1.
    assert by_text["Assets rose."] == f"{fin} > Consolidated balance sheets analysis"
    assert by_text["Sales grew."] == "Part I > Item 2. Management's Discussion and Analysis"


def test_title_anchors_do_not_override_real_items():
    items = [
        _para("Item 7. MD&A", 1, strong=True),
        _para("Consolidated Statements of Operations", 1, strong=True),
        _para("Revenue was 10.", 1),
    ]
    blocks = build_blocks(items, "10-K", n_pages=1)
    assert blocks[-1].section == "Item 7. MD&A > Consolidated Statements of Operations"


def test_running_header_anchor_after_part_iv():
    items = [_para("Item 15. Exhibits", 1, strong=True), _para("See the index.", 1)]
    for page in range(2, 6):
        items += [
            _para("Management’s discussion and analysis", page, edge=True),
            _para(f"MD&A text on page {page}.", page),
            _para(f"JPMorgan Chase & Co./2025 Form 10-K {page}", page, edge=True),
        ]
    for page in range(6, 10):
        items += [
            _para("Notes to consolidated financial statements", page, edge=True),
            _para(f"Note text on page {page}.", page),
        ]
    blocks = build_blocks(items, "10-K", n_pages=9)
    sec = {b.text: b.section for b in blocks}
    assert sec["MD&A text on page 2."].startswith("Item 7. Management’s Discussion")
    assert sec["MD&A text on page 5."].startswith("Item 7. Management’s Discussion")
    assert sec["Note text on page 7."] == "Item 8. Financial Statements and Supplementary Data"
    assert not any("Form 10-K" in b.text for b in blocks)  # footer still dropped


def test_toc_page_numbers_place_missing_items():
    toc = (
        "|  | Part I – Financial information | Page |\n|---|---|---|\n"
        "| Item 1. | Financial Statements |  |\n"
        "|  | Consolidated statements of income | 6 |\n"
        "| Item 2. | Management's Discussion and Analysis |  |\n"
        "|  | Executive Overview | 3 |\n"
        "| Item 3. | Market Risk | 8 |"
    )
    items = [RawItem("table", toc, 1, table_id="t1")]
    for page in range(2, 10):
        items += [
            _para(f"Body text on page {page}.", page),
            _para(f"ACME Q2 Form 10-Q {page - 1}", page, edge=True),  # printed = physical - 1
        ]
    blocks = build_blocks(items, "10-Q", n_pages=9)
    sec = {b.text: b.section for b in blocks}
    assert sec["Body text on page 3."] == "Cover"
    assert sec["Body text on page 4."] == "Part I > Item 2. Management's Discussion and Analysis"
    assert sec["Body text on page 7."] == "Part I > Item 1. Financial Statements"
    assert sec["Body text on page 9."] == "Part I > Item 3. Market Risk"


def test_repeated_part_header_does_not_reset_item():
    items = [
        _para("PART I", 1, strong=True),
        _para("Item 1. Business", 1, strong=True),
        _para("Text one.", 1),
        _para("PART I", 2, strong=True),
        _para("Text two.", 2),
    ]
    blocks = build_blocks(items, "10-K", n_pages=2)
    assert blocks[-1].section == "Item 1. Business"


def test_letter_spaced_pdf_headings_are_repaired():
    from ledgerlens.ingest.sections import repair_spacing

    assert repair_spacing("Item 1. Busines s") == "Item 1. Business"
    assert repair_spacing("PART II I") == "PART III"
    assert repair_spacing("CRITICAL ACCOUNTING ESTIMA TES") == "CRITICAL ACCOUNTING ESTIMATES"
    assert repair_spacing("Cash and Cash Equivalents") == "Cash and Cash Equivalents"
    assert repair_spacing("PART I") == "PART I"
    items = [_para("ITEM 1. BUSINES S", 1), _para("Text.", 1)]
    pdf = build_blocks(items, "10-K", n_pages=1, repair_spacing_headings=True)
    assert pdf[0].text == "Item 1. Business"


def test_large_font_layout_table_title_is_a_heading(html_ref):
    html = (
        b"<html><body>"
        b"<div style='font-size:9pt'>Body text sets the normal size for this filing.</div>"
        b"<div style='font-size:9pt'>More body text so the body size is clear.</div>"
        b"<table><tr><td><span style='font-size:18pt;font-weight:400'>"
        b"Management's Discussion and Analysis</span></td></tr></table>"
        b"<div style='font-size:9pt'>Revenue grew in every segment this year.</div>"
        b"</body></html>"
    )
    parsed = parse_html(html, html_ref)
    revenue = next(b for b in parsed.blocks if b.text.startswith("Revenue grew"))
    assert revenue.section.startswith("Item 7. Management’s Discussion")
