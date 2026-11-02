"""Parse EDGAR 10-K / 10-Q HTML (usually inline XBRL) into Blocks.

The walk flattens the DOM into paragraphs and tables in reading order. The hidden
`ix:header` (XBRL contexts and units) and `display:none` content are dropped. Page
numbers are estimated by counting page breaks: `page-break-before/after` and
`break-before/after` styles, and `<hr>` page separators.
"""

from __future__ import annotations

import re
from typing import Any

import lxml.html

from ledgerlens.ingest.sections import RawItem, build_blocks, clean_text
from ledgerlens.ingest.tables import rows_to_markdown
from ledgerlens.logging import get_logger
from ledgerlens.schemas import FilingRef, ParsedFiling

log = get_logger(__name__)

_SKIP_TAGS = {"script", "style", "head", "title", "noscript", "ix:header", "meta", "link"}
_BLOCK_TAGS = {
    "p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "body", "html",
    "center", "blockquote", "section", "article", "pre", "dl", "dt", "dd"
}  # fmt: skip
_BOLD_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "b", "strong"}
_BOLD_RE = re.compile(r"font-weight:\s*(bold|[6-9]00)")
_BREAK_BEFORE = re.compile(r"(page-break-before|break-before):\s*(always|page)")
_BREAK_AFTER = re.compile(r"(page-break-after|break-after):\s*(always|page)")
_HIDDEN = re.compile(r"display:\s*none")
_FONT_SIZE = re.compile(r"font-size:\s*([\d.]+)\s*(pt|px)")
LARGE_FONT_RATIO = 1.2
MAX_LARGE_HEADING_LEN = 160


def _font_size(style: str) -> float | None:
    m = _FONT_SIZE.search(style)
    if not m:
        return None
    size = float(m.group(1))
    return size * 0.75 if m.group(2) == "px" else size


class _Walker:
    """Flatten the DOM into RawItems while counting page breaks."""

    def __init__(self, table_prefix: str) -> None:
        self.items: list[RawItem] = []
        self.page = 1
        self.pieces: list[tuple[str, bool, float | None]] = []
        self.sizes: dict[int, float] = {}  # item index -> largest font size in it
        self.chars_by_size: dict[float, int] = {}
        self.para_page = 1
        self.table_prefix = table_prefix
        self.n_tables = 0
        self.page_start_index = 0

    # ------------------------------------------------------------ paragraphs

    def add_text(self, text: str | None, bold: bool, size: float | None = None) -> None:
        if not text:
            return
        if not self.pieces or not any(t.strip() for t, _, _ in self.pieces):
            self.para_page = self.page
        self.pieces.append((text, bold, size))

    def flush(self) -> None:
        if not self.pieces:
            return
        text = clean_text("".join(t for t, _, _ in self.pieces))
        if text:
            strong = all(b for t, b, _ in self.pieces if t.strip())
            sizes = [z for t, _, z in self.pieces if t.strip() and z is not None]
            if sizes:
                self.sizes[len(self.items)] = max(sizes)
                for t, _, z in self.pieces:
                    if z is not None:
                        self.chars_by_size[z] = self.chars_by_size.get(z, 0) + len(t.strip())
            self.items.append(RawItem(kind="para", text=text, page=self.para_page, strong=strong))
        self.pieces = []

    def page_break(self) -> None:
        self.flush()
        self._mark_edges()
        self.page += 1

    def _mark_edges(self) -> None:
        """Flag the first two and last three items on the page just finished."""
        page_items = self.items[self.page_start_index :]
        for it in page_items[:2] + page_items[-3:]:
            it.edge = True
        self.page_start_index = len(self.items)

    # ------------------------------------------------------------ tables

    def add_table(self, el: Any, bold: bool = False, size: float | None = None) -> None:
        self.flush()
        rows = _table_rows(el)
        md = rows_to_markdown(rows)
        if md is None:
            # A layout grid: read it as prose, one paragraph per row, keeping the
            # styling so large or bold titles set in layout tables still count.
            for tr in el.iter("tr"):
                if _nearest_table(tr) is not el:
                    continue
                for cell in tr:
                    if isinstance(cell.tag, str) and cell.tag.lower() in ("td", "th"):
                        self.walk(cell, bold, size)
                        self.add_text(" ", bold, size)
                self.flush()
            return
        self.n_tables += 1
        self.items.append(
            RawItem(
                kind="table",
                text=md,
                page=self.page,
                table_id=f"{self.table_prefix}:table:{self.n_tables:04d}",
            )
        )

    # ------------------------------------------------------------ walk

    def walk(self, el: Any, bold: bool, size: float | None = None) -> None:
        tag = el.tag if isinstance(el.tag, str) else ""
        tag = tag.lower()
        if not tag or tag in _SKIP_TAGS:
            return
        style = (el.get("style") or "").lower()
        if _HIDDEN.search(style):
            return
        if tag == "hr":
            self.page_break()
            return
        if _BREAK_BEFORE.search(style):
            self.page_break()
        if tag == "table":
            self.add_table(el, bold, _font_size(style) or size)
        else:
            is_block = tag in _BLOCK_TAGS
            if is_block:
                self.flush()
            here_bold = bold or tag in _BOLD_TAGS or bool(_BOLD_RE.search(style))
            here_size = _font_size(style) or size
            if tag == "br":
                self.add_text(" ", here_bold, here_size)
            self.add_text(el.text, here_bold, here_size)
            for child in el:
                self.walk(child, here_bold, here_size)
                self.add_text(child.tail, here_bold, here_size)
            if is_block:
                self.flush()
        if _BREAK_AFTER.search(style):
            self.page_break()

    def finish(self) -> list[RawItem]:
        self.flush()
        self._mark_edges()
        # Short paragraphs set clearly larger than body text are heading candidates too
        # (Intel styles its headings by size and colour, not weight).
        if self.chars_by_size:
            body = max(self.chars_by_size, key=lambda z: self.chars_by_size[z])
            for idx, z in self.sizes.items():
                it = self.items[idx]
                if z >= body * LARGE_FONT_RATIO and len(it.text) <= MAX_LARGE_HEADING_LEN:
                    it.strong = True
        return self.items


def _table_rows(table: Any) -> list[list[str]]:
    """Expand a table into a grid. Spanned text goes in the first column of its span."""
    rows: list[list[str]] = []
    for tr in table.iter("tr"):
        # Skip rows of nested tables; they are rare in EDGAR and flattened as text.
        if _nearest_table(tr) is not table:
            continue
        row: list[str] = []
        for cell in tr:
            if not isinstance(cell.tag, str) or cell.tag.lower() not in ("td", "th"):
                continue
            text = clean_text(" ".join(cell.itertext()))
            try:
                span = max(1, min(int(cell.get("colspan", "1")), 50))
            except ValueError:
                span = 1
            row.append(text)
            row.extend([""] * (span - 1))
        rows.append(row)
    return rows


def _nearest_table(el: Any) -> Any:
    parent = el.getparent()
    while parent is not None and parent.tag != "table":
        parent = parent.getparent()
    return parent


def parse_html(raw: bytes, filing: FilingRef) -> ParsedFiling:
    """Parse one EDGAR HTML document into a ParsedFiling."""
    doc = lxml.html.document_fromstring(raw)
    walker = _Walker(table_prefix=filing.filing_id)
    body = doc.find("body")
    walker.walk(body if body is not None else doc, bold=False)
    items = walker.finish()
    blocks = build_blocks(items, filing.form, n_pages=walker.page)
    log.debug("parsed_html", filing_id=filing.filing_id, pages=walker.page, blocks=len(blocks))
    return ParsedFiling(filing=filing, blocks=blocks)
