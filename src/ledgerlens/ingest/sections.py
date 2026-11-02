"""Shared section logic for HTML and PDF parsers.

Parsers emit `RawItem`s (paragraphs and tables in reading order, with a page and a
"strong" flag for bold or large text). `build_blocks` then drops repeated page
headers and footers, classifies headings (Part, Item, bold sub-headings), suppresses
table-of-contents runs and tracks the section path for every block.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ledgerlens.schemas import Block

ITEM_RE = re.compile(r"^item\s*(\d{1,2}[a-d]?)(?![a-z0-9])\s*[.:\-–—]?\s*(.*)$", re.IGNORECASE)
PART_RE = re.compile(r"^part\s+(iv|i{1,3})\b\s*[.:\-–—]?\s*(.*)$", re.IGNORECASE)
_PAGE_NO = re.compile(
    r"^(page\s*)?[-–—\s]*([0-9]{1,3}|[ivxlc]{1,6})[-–—\s]*$|^[A-Z]?-?\d{1,3}$", re.IGNORECASE
)
_TRAILING_PAGE_REF = re.compile(r"\s\d{1,3}$")
_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")

MAX_ITEM_LEN = 160
MAX_SUBHEADING_LEN = 120
MAX_SUBHEADING_WORDS = 15
COVER = "Cover"


@dataclass
class RawItem:
    """One paragraph or table in reading order, before section assignment."""

    kind: str  # "para" or "table"
    text: str
    page: int
    strong: bool = False
    edge: bool = False  # first/last items of a page: header and footer candidates
    table_id: str | None = None


def clean_text(text: str) -> str:
    return _WS.sub(" ", text.replace("\xa0", " ").replace("​", "")).strip()


def _title(text: str) -> str:
    text = text.strip(" .:-–—")
    if text.isupper():
        text = " ".join(_cap_word(w) for w in text.split())
    return text


def _cap_word(word: str) -> str:
    """Title-case one word, leaving abbreviations (D.C., MD&A) and codes (10-K) alone."""
    if "." in word.strip(".") or any(ch.isdigit() or ch in "&/" for ch in word):
        return word
    return word[:1].upper() + word[1:].lower()


def match_item(text: str) -> tuple[str, str] | None:
    """Return (item number, title) for an `Item 7. MD&A` style heading line."""
    if len(text) > MAX_ITEM_LEN:
        return None
    m = ITEM_RE.match(text)
    if not m:
        return None
    number = m.group(1).upper()
    title = m.group(2).strip()
    if title and title[0].islower():  # "Item 7 of this report ..." is a cross reference
        return None
    if _TRAILING_PAGE_REF.search(title) or re.fullmatch(r"\d{1,3}", title):
        return None  # table-of-contents line with a page number
    return number, _title(title)


def match_part(text: str) -> tuple[str, str] | None:
    if len(text) > 100:
        return None
    m = PART_RE.match(text)
    if not m:
        return None
    title = m.group(2).strip()
    if title and title[0].islower():
        return None
    if _TRAILING_PAGE_REF.search(title):
        return None
    return m.group(1).upper(), _title(title)


def is_page_number(text: str) -> bool:
    return bool(_PAGE_NO.match(text.strip()))


def is_subheading(text: str) -> bool:
    if len(text) > MAX_SUBHEADING_LEN or len(text.split()) > MAX_SUBHEADING_WORDS:
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    if is_page_number(text):
        return False
    # Sentences ending in a full stop are bold emphasis, not headings.
    return not (text.endswith(".") and len(text.split()) > 6)


def drop_headers_footers(items: list[RawItem], n_pages: int) -> list[RawItem]:
    """Remove page numbers and text repeated at the top or bottom of many pages."""
    keyed = Counter(
        _DIGITS.sub("#", it.text.lower())
        for it in items
        if it.kind == "para" and it.edge and len(it.text) <= 100
    )
    threshold = max(3, int(0.1 * n_pages))
    repeated = {k for k, v in keyed.items() if v >= threshold}
    out = []
    for it in items:
        if it.kind == "para" and it.edge:
            if is_page_number(it.text):
                continue
            if len(it.text) <= 100 and _DIGITS.sub("#", it.text.lower()) in repeated:
                continue
        out.append(it)
    return out


@dataclass
class _Heading:
    level: str  # "part", "item" or "sub"
    label: str


def _classify(items: list[RawItem]) -> list[_Heading | None]:
    """Decide which paragraphs are headings. Item titles split over two lines are joined."""
    heads: list[_Heading | None] = [None] * len(items)
    skip: set[int] = set()
    for i, it in enumerate(items):
        if it.kind != "para" or i in skip:
            continue
        text = it.text
        if (m := match_item(text)) and (it.strong or len(text) <= 100):
            number, title = m
            if not title:
                nxt = i + 1
                if nxt < len(items) and items[nxt].kind == "para":
                    cand = items[nxt].text
                    if (
                        len(cand) <= 150
                        and not match_item(cand)
                        and not match_part(cand)
                        and not _TRAILING_PAGE_REF.search(cand)
                    ):
                        title = _title(cand)
                        skip.add(nxt)
            label = f"Item {number}. {title}" if title else f"Item {number}."
            heads[i] = _Heading("item", label)
        elif (m := match_part(text)) and (it.strong or len(text) <= 60):
            number, title = m
            heads[i] = _Heading("part", f"Part {number}" + (f". {title}" if title else ""))
        elif it.strong and is_subheading(text):
            heads[i] = _Heading("sub", _title(text) if text.isupper() else text.strip())
    # A title consumed by the line above is folded into that heading.
    for i in skip:
        heads[i] = _Heading("merged", "")
    return heads


def _suppress_toc(items: list[RawItem], heads: list[_Heading | None]) -> None:
    """Demote runs of 3+ Item headings with no content between them (a text TOC).

    Page numbers and short lines ending in a page reference count as TOC filler, not
    content. A Part heading ends the run.
    """
    run: list[int] = []

    def flush() -> None:
        if len(run) >= 3:
            for idx in run:
                heads[idx] = None
                nxt = heads[idx + 1] if idx + 1 < len(heads) else None
                if nxt is not None and nxt.level == "merged":
                    heads[idx + 1] = None
        run.clear()

    def toc_filler(it: RawItem, h: _Heading | None) -> bool:
        if h is not None:
            return h.level == "merged"
        if it.kind != "para":
            return False
        short_ref = len(it.text) <= 150 and bool(_TRAILING_PAGE_REF.search(it.text))
        return len(it.text) <= 3 or is_page_number(it.text) or short_ref

    for i, it in enumerate(items):
        h = heads[i]
        if h is not None and h.level == "item":
            run.append(i)
        elif h is not None and h.level == "part":
            flush()  # a real Part heading ends any TOC run before it
        elif toc_filler(it, h):
            continue
        else:
            flush()
    flush()


def build_blocks(items: list[RawItem], form: str, n_pages: int) -> list[Block]:
    """Assign headings and section paths and return Blocks in reading order."""
    items = drop_headers_footers(items, n_pages)
    heads = _classify(items)
    _suppress_toc(items, heads)

    part: str | None = None
    item: str | None = None
    sub: str | None = None
    include_part = form == "10-Q"

    def path() -> str:
        parts: list[str] = []
        if part and (include_part or item is None):
            parts.append(part)
        if item:
            parts.append(item)
        if sub:
            parts.append(sub)
        return " > ".join(parts) if parts else COVER

    blocks: list[Block] = []
    for it, h in zip(items, heads, strict=True):
        if h is not None and h.level == "merged":
            continue
        if h is not None:
            if h.level == "part":
                part, item, sub = h.label.split(".")[0], None, None
            elif h.level == "item":
                item, sub = h.label, None
            else:
                sub = h.label
            blocks.append(Block(kind="heading", text=h.label, section=path(), page=it.page))
            continue
        if it.kind == "table":
            blocks.append(
                Block(
                    kind="table", text=it.text, section=path(), page=it.page, table_id=it.table_id
                )
            )
        elif it.text:
            blocks.append(Block(kind="text", text=it.text, section=path(), page=it.page))
    return blocks


def top_level_section(section: str) -> str:
    """The Item (or Part, or Cover) a section path belongs to."""
    parts = section.split(" > ")
    for p in parts:
        if p.startswith("Item "):
            return p
    return parts[0]


def item_number(section: str) -> str | None:
    """`Item 7. MD&A > Liquidity` -> `7`."""
    for p in section.split(" > "):
        m = re.match(r"Item (\d{1,2}[A-D]?)\.", p)
        if m:
            return m.group(1)
    return None
