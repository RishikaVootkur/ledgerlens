import pymupdf
import pytest

from ledgerlens.ingest.parse_pdf import parse_pdf


def _draw_table(page, x0, y0, rows, col_w=(160, 80, 80), row_h=18):
    width = sum(col_w)
    for r in range(len(rows) + 1):
        y = y0 + r * row_h
        page.draw_line((x0, y), (x0 + width, y))
    x = x0
    for w in (0, *col_w):
        x += w
        page.draw_line((x, y0), (x, y0 + len(rows) * row_h))
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            page.insert_text((x + 3, y0 + r * row_h + 13), cell, fontsize=9)
            x += col_w[c]


@pytest.fixture
def sample_pdf() -> bytes:
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 80), "Item 1. Business", fontsize=14, fontname="hebo")
    p1.insert_text(
        (72, 110), "Sample Corp makes widgets for customers around the world.", fontsize=10
    )
    p1.insert_text((300, 800), "1", fontsize=9)
    p2 = doc.new_page()
    p2.insert_text(
        (72, 80), "Item 7. Management's Discussion and Analysis", fontsize=14, fontname="hebo"
    )
    p2.insert_text((72, 110), "Revenue grew on strong widget demand this year.", fontsize=10)
    _draw_table(
        p2,
        72,
        140,
        [["", "2025", "2024"], ["Net sales", "$1,234", "$1,100"], ["Costs", "(321)", "45"]],
    )
    p2.insert_text((300, 800), "2", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return bytes(data)


def test_pdf_pages_are_exact_and_headings_found(sample_pdf, pdf_ref):
    parsed = parse_pdf(sample_pdf, pdf_ref)
    by_text = {b.text: b for b in parsed.blocks}
    assert by_text["Item 1. Business"].kind == "heading"
    assert by_text["Item 1. Business"].page == 1
    item7 = "Item 7. Management's Discussion and Analysis"
    assert by_text[item7].kind == "heading"
    assert by_text[item7].page == 2
    revenue = next(b for b in parsed.blocks if b.text.startswith("Revenue grew"))
    assert revenue.page == 2
    assert revenue.section == item7


def test_pdf_page_numbers_dropped(sample_pdf, pdf_ref):
    parsed = parse_pdf(sample_pdf, pdf_ref)
    assert not any(b.text in {"1", "2"} for b in parsed.blocks)


def test_pdf_table_to_markdown(sample_pdf, pdf_ref):
    parsed = parse_pdf(sample_pdf, pdf_ref)
    tables = [b for b in parsed.blocks if b.kind == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert t.page == 2
    assert t.table_id == "FB-SAMPLE_2025_10K:table:0001"
    assert t.text.splitlines() == [
        "|  | 2025 | 2024 |",
        "|---|---|---|",
        "| Net sales | $1,234 | $1,100 |",
        "| Costs | (321) | 45 |",
    ]
    # Table words are not repeated as loose text.
    assert not any(b.kind == "text" and "Net sales" in b.text for b in parsed.blocks)


def _span(text, bbox, size=10.0, flags=0):
    return {"text": text, "bbox": bbox, "size": size, "flags": flags, "font": "Helvetica"}


def test_overlap_ratio_uses_area_not_centre():
    from ledgerlens.ingest.parse_pdf import overlap_ratio

    table = [(100.0, 100.0, 300.0, 200.0)]
    assert overlap_ratio((150, 120, 200, 130), table) == 1.0
    assert overlap_ratio((10, 10, 50, 20), table) == 0.0
    # Centre outside the table but 60% of the span inside it.
    assert overlap_ratio((60, 150, 160, 160), table) == pytest.approx(0.6)


def test_text_spans_inside_a_table_are_dropped():
    from ledgerlens.ingest.parse_pdf import _block_items

    table = [(100.0, 100.0, 300.0, 200.0)]
    block = {
        "bbox": (40, 90, 300, 220),
        "lines": [
            {
                "bbox": (40, 90, 300, 100),
                "spans": [_span("Net income rose sharply.", (40, 90, 95, 100))],
            },
            {
                "bbox": (60, 150, 300, 160),
                "spans": [
                    _span("$ 4,869", (60, 150, 160, 160)),
                    _span("5,058", (200, 150, 240, 160)),
                ],
            },
            {
                "bbox": (40, 210, 300, 220),
                "spans": [_span("Cash grew too.", (40, 210, 90, 220))],
            },
        ],
    }
    items = _block_items(block, 1, False, 10.0, table)
    texts = " ".join(i[2].text for i in items)
    assert "Net income rose sharply." in texts
    assert "Cash grew too." in texts
    assert "4,869" not in texts
    assert "5,058" not in texts
