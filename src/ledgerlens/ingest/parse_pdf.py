"""Parse filing PDFs (FinanceBench documents) into Blocks with exact 1-based pages.

Tables come from `page.find_tables()`. Cell text is rebuilt from the page's words
inside each cell box, because `Table.extract()` drops spaces and splits numbers on
many SEC PDFs. Text outside tables is read block by block; lines that are bold,
larger than body text, or match `Item 7.` / `Part II` patterns become heading
candidates for the shared section logic.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pymupdf

from ledgerlens.ingest.sections import (
    RawItem,
    build_blocks,
    clean_text,
    looks_like_column_header,
    match_item,
    match_part,
)
from ledgerlens.ingest.tables import rows_to_markdown
from ledgerlens.logging import get_logger
from ledgerlens.schemas import FilingRef, ParsedFiling

log = get_logger(__name__)

EDGE_FRACTION = 0.08
_BOLD_FLAG = 16


def _span_bold(span: dict[str, Any]) -> bool:
    font = str(span.get("font", "")).lower()
    return bool(int(span.get("flags", 0)) & _BOLD_FLAG) or "bold" in font or "black" in font


def _body_font_size(pages: list[dict[str, Any]]) -> float:
    sizes: Counter[float] = Counter()
    for page in pages:
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(float(span["size"]) * 2) / 2] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _cell_text(words: list[tuple[Any, ...]], bbox: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bbox
    inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    inside.sort(key=lambda w: (round(w[3]), w[0]))
    return " ".join(str(w[4]) for w in inside)


def _page_tables(
    page: Any, words: list[tuple[Any, ...]]
) -> list[tuple[tuple[float, float, float, float], list[list[str]]]]:
    try:
        found = page.find_tables()
    except Exception as exc:  # PyMuPDF raises on some malformed pages
        log.warning("find_tables_failed", page=page.number + 1, error=str(exc))
        return []
    out = []
    for table in found.tables:
        rows: list[list[str]] = []
        for row in table.rows:
            rows.append(["" if c is None else _cell_text(words, c) for c in row.cells])
        out.append((tuple(table.bbox), rows))
    return out


Box = tuple[float, float, float, float]
TABLE_OVERLAP = 0.5


def overlap_ratio(bbox: tuple[float, ...], boxes: list[Box]) -> float:
    """Largest share of `bbox`'s area that lies inside any of `boxes`."""
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    if area <= 0:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return 1.0 if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes) else 0.0
    best = 0.0
    for b in boxes:
        w = min(x1, b[2]) - max(x0, b[0])
        h = min(y1, b[3]) - max(y0, b[1])
        if w > 0 and h > 0:
            best = max(best, w * h / area)
    return best


def _join_lines(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        if out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}" if out else ln
    return clean_text(out)


def _block_items(
    block: dict[str, Any],
    page_no: int,
    edge: bool,
    body_size: float,
    table_boxes: list[Box] | None = None,
) -> list[tuple[float, float, RawItem]]:
    """Split one text block into paragraphs, cutting at heading-like lines.

    Spans that lie mostly inside a table box are dropped: the table already has them.
    """
    boxes = table_boxes or []
    out: list[tuple[float, float, RawItem]] = []
    x = float(block["bbox"][0])
    pending: list[str] = []
    first_y = float(block["bbox"][1])

    def flush() -> None:
        if pending:
            text = _join_lines(pending)
            if text:
                out.append((first_y, x, RawItem("para", text, page_no, edge=edge)))
            pending.clear()

    for line in block.get("lines", []):
        kept = [
            s
            for s in line["spans"]
            if not boxes or overlap_ratio(tuple(s["bbox"]), boxes) <= TABLE_OVERLAP
        ]
        spans = [s for s in kept if s["text"].strip()]
        if not spans:
            continue
        text = clean_text("".join(s["text"] for s in kept))
        ly = float(line["bbox"][1])
        size = max(float(s["size"]) for s in spans)
        bold = all(_span_bold(s) for s in spans)
        large = size >= body_size * 1.2
        keyword = bool(match_item(text) or match_part(text))
        if (bold or large or keyword) and len(text) <= 160:
            flush()
            strong = bold or large
            out.append((ly, float(line["bbox"][0]), RawItem("para", text, page_no, strong, edge)))
        else:
            if not pending:
                first_y = ly
            pending.append(text)
    flush()
    return out


def _open_pdf(raw: bytes) -> Any:
    opener: Any = pymupdf.open  # PyMuPDF ships partial annotations
    return opener(stream=raw, filetype="pdf")


def parse_pdf(raw: bytes, filing: FilingRef) -> ParsedFiling:
    """Parse one PDF into a ParsedFiling. Page numbers are the PDF's physical pages."""
    doc = _open_pdf(raw)
    try:
        dicts = [page.get_text("dict", sort=True) for page in doc]
        body_size = _body_font_size(dicts)
        items: list[RawItem] = []
        n_tables = 0
        for page_no, (page, pdict) in enumerate(zip(doc, dicts, strict=True), start=1):
            height = float(page.rect.height) or 1.0
            words = page.get_text("words")
            # (y, x, item) so tables and text interleave in reading order.
            placed: list[tuple[float, float, RawItem]] = []
            table_boxes: list[Box] = []
            for bbox, rows in _page_tables(page, words):
                md = rows_to_markdown(rows)
                if md is None:
                    continue  # layout grid: its words are read as normal text below
                n_tables += 1
                table_boxes.append(bbox)
                tid = f"{filing.filing_id}:table:{n_tables:04d}"
                placed.append((bbox[1], bbox[0], RawItem("table", md, page_no, table_id=tid)))
            for block in pdict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                y0, y1 = float(block["bbox"][1]), float(block["bbox"][3])
                edge = y1 < height * EDGE_FRACTION or y0 > height * (1 - EDGE_FRACTION)
                placed.extend(_block_items(block, page_no, edge, body_size, table_boxes))
            placed.sort(key=lambda t: (round(t[0]), t[1]))
            page_items = [p[2] for p in placed]
            # Short text in the first or last slots of a page is a header/footer candidate.
            for it in page_items[:2] + page_items[-3:]:
                if it.kind == "para" and len(it.text) <= 60:
                    it.edge = True
            items.extend(page_items)
        blocks = build_blocks(
            _merge_strong_runs(items),
            filing.form,
            n_pages=doc.page_count,
            repair_spacing_headings=True,
        )
        log.debug(
            "parsed_pdf", filing_id=filing.filing_id, pages=doc.page_count, blocks=len(blocks)
        )
        return ParsedFiling(filing=filing, blocks=blocks)
    finally:
        doc.close()


def _merge_strong_runs(items: list[RawItem]) -> list[RawItem]:
    """Join consecutive bold lines on the same page into one heading candidate.

    Multi-line bold headings ("Consolidated Statements of" / "Cash Flows") would
    otherwise become two headings. Item and Part lines are never merged.
    """
    out: list[RawItem] = []
    for it in items:
        prev = out[-1] if out else None
        if (
            prev is not None
            and it.kind == "para"
            and prev.kind == "para"
            and it.strong
            and prev.strong
            and prev.page == it.page
            and not it.edge
            and not prev.edge
            and not (match_item(it.text) or match_part(it.text))
            and not (match_item(prev.text) or match_part(prev.text))
            and len(prev.text) + len(it.text) <= 120
            and not looks_like_column_header(it.text)
        ):
            prev.text = f"{prev.text} {it.text}"
            continue
        out.append(it)
    return out
