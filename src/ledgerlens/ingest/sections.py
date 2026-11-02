"""Shared section logic for HTML and PDF parsers.

Parsers emit `RawItem`s (paragraphs and tables in reading order, with a page and a
"strong" flag for bold or large text). `build_blocks` then drops repeated page
headers and footers, classifies headings (Part, Item, bold sub-headings), suppresses
table-of-contents runs, maps well-known titles and TOC page numbers to Items for
filers without Item headings, and tracks the section path for every block. Headings
seen before any Part or Item nest under `Cover`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ledgerlens.ingest.tables import markdown_table_rows
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

    kind: str  # "para", "table" or "anchor" (a running header naming a known section)
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


def looks_like_column_header(text: str) -> bool:
    """Bold table column lines such as `(Millions) 2018 2017 2016` are not headings."""
    words = text.split()
    numeric = sum(1 for w in words if any(ch.isdigit() for ch in w))
    return text.startswith("(") or numeric * 2 >= len(words)


def is_subheading(text: str) -> bool:
    if len(text) > MAX_SUBHEADING_LEN or len(text.split()) > MAX_SUBHEADING_WORDS:
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    if is_page_number(text) or looks_like_column_header(text):
        return False
    # Sentences ending in a full stop are bold emphasis, not headings.
    return not (text.endswith(".") and len(text.split()) > 6)


# ---------------------------------------------------------------- spacing repair

_STOPWORDS = {"a", "an", "as", "at", "be", "by", "in", "is", "of", "on", "or", "to", "and", "the",
              "for", "vs", "per", "its", "our", "net", "new", "non", "end"}  # fmt: skip
_FRAGMENT = re.compile(r"(?<=[A-Za-z]{3}) ([a-z]{1,3})(?=[\s,.:;)]|$)")
_FRAGMENT_UPPER = re.compile(r"(?<=[A-Z]{3}) ([A-Z]{1,3})(?=[\s,.:;)]|$)")
_KEEP_UPPER = {w.upper() for w in _STOPWORDS} | {"I", "II", "III", "IV", "V", "US", "USA", "UK"}
_ROMAN_GAP = re.compile(r"\b(part\s+)(i{1,2}) (i{1,2}|v)\b", re.IGNORECASE)


def repair_spacing(text: str) -> str:
    """Rejoin letter-spaced PDF words in heading candidates ("Busines s" -> "Business").

    Only a trailing fragment of one to three lowercase letters that is not a common
    short word is joined, and "PART II I" becomes "PART III".
    """
    text = _ROMAN_GAP.sub(lambda m: m.group(1) + m.group(2) + m.group(3), text)
    if text.isupper():
        return _FRAGMENT_UPPER.sub(
            lambda m: m.group(0) if m.group(1) in _KEEP_UPPER else m.group(1), text
        )
    return _FRAGMENT.sub(lambda m: m.group(0) if m.group(1) in _STOPWORDS else m.group(1), text)


# ---------------------------------------------------------------- anchors
#
# Some filers (JPMorgan, Intel, Linde) do not put "Item 7." style headings over the
# content: the Items live in a cross-reference index and the body uses the annual
# report's own headings. Anchors map well-known titles, running page headers and
# table-of-contents page numbers back to their Item, so section paths, filters and
# the quality check still work. Title and running-header anchors only apply where no
# real Item heading is in force: before the first Item, after Item 15/16 of a 10-K,
# or after another anchor.

_ANCHOR_PATTERNS: dict[str, re.Pattern[str]] = {
    "mda": re.compile(
        r"management'?s discussion and analysis"
        r"( of (the )?financial condition and results of operations)?( \(md&a\))?"
    ),
    "market_risk": re.compile(
        r"quantitative and qualitative disclosures? (about|of) market risks?"
    ),
    "financials": re.compile(
        r"(condensed )?consolidated (condensed )?(financial statements|statements? of "
        r"(income|operations|earnings|comprehensive (income|loss)|cash flows"
        r"|financial (position|condition)"
        r"|changes in (stockholders|shareholders|share ?owners)'? equity)|balance sheets?)"
        r"|notes to (the )?(condensed )?consolidated (condensed )?financial statements"
        r"|report of independent registered public accounting firm"
        r"|financial statements and supplement(al|ary) (details|data|information)"
    ),
    "controls": re.compile(r"controls and procedures"),
    "risk_factors": re.compile(r"risk factors( and other key information)?"),
}

_MDA_TITLE = "Management’s Discussion and Analysis of Financial Condition and Results of Operations"
_ANCHOR_10K = {
    "mda": f"Item 7. {_MDA_TITLE}",
    "market_risk": "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
    "financials": "Item 8. Financial Statements and Supplementary Data",
    "controls": "Item 9A. Controls and Procedures",
    "risk_factors": "Item 1A. Risk Factors",
}
_ANCHOR_10Q = {
    "mda": ("Part I", f"Item 2. {_MDA_TITLE}"),
    "market_risk": ("Part I", "Item 3. Quantitative and Qualitative Disclosures About Market Risk"),
    "financials": ("Part I", "Item 1. Financial Statements"),
    "controls": ("Part I", "Item 4. Controls and Procedures"),
    "risk_factors": ("Part II", "Item 1A. Risk Factors"),
}


@dataclass
class _Heading:
    level: str  # "part", "item", "sub", "merged" or "anchor"
    label: str
    part: str | None = None  # anchors only: the Part an anchored 10-Q item belongs to


def anchor_key(text: str) -> str | None:
    """Which well-known section a title names, if any (full match only)."""
    t = _WS.sub(" ", text.lower().replace("’", "'").replace("‘", "'"))
    t = re.sub(r"\s*\((unaudited|continued)\)\s*$", "", t).strip(" .:")
    for key, pattern in _ANCHOR_PATTERNS.items():
        if pattern.fullmatch(t):
            return key
    return None


def _anchor_heading(key: str, form: str) -> _Heading | None:
    if form == "10-K":
        return _Heading("anchor", _ANCHOR_10K[key])
    if form == "10-Q":
        part, label = _ANCHOR_10Q[key]
        return _Heading("anchor", label, part=part)
    return None


# ---------------------------------------------------------------- printed pages + TOC

_TRAILING_NUMBER = re.compile(r"(\d{1,3})\s*$")
_TOC_ITEM = re.compile(r"^item\s*(\d{1,2}[a-d]?)\b\.?\s*(.*)$", re.IGNORECASE)
_TOC_PART = re.compile(r"(?:^|\s)part\s+(iv|i{1,3})\b", re.IGNORECASE)
_INT = re.compile(r"\b(\d{1,3})\b")


def _repeated_edge_keys(items: list[RawItem], n_pages: int) -> set[str]:
    keyed = Counter(
        _DIGITS.sub("#", it.text.lower())
        for it in items
        if it.kind == "para" and it.edge and len(it.text) <= 100
    )
    threshold = max(3, int(0.1 * n_pages))
    return {k for k, v in keyed.items() if v >= threshold}


def printed_pages(items: list[RawItem], repeated: set[str]) -> dict[int, int]:
    """Physical page -> printed page number, read from page footers and headers."""
    out: dict[int, int] = {}
    for it in items:
        if it.kind != "para" or not it.edge or it.page in out or len(it.text) > 100:
            continue
        text = it.text.strip()
        if text.isdigit():
            out[it.page] = int(text)
        elif _DIGITS.sub("#", text.lower()) in repeated and (m := _TRAILING_NUMBER.search(text)):
            out[it.page] = int(m.group(1))
    return out


@dataclass
class TocEntry:
    part: str | None
    number: str
    title: str
    printed_page: int


def parse_toc(tables: list[str]) -> list[TocEntry]:
    """Item rows and their printed start pages from table-of-contents tables.

    An Item row without a page takes the first page listed under it (JPMorgan lists
    the page of each sub-entry, not of the Item).
    """
    entries: list[TocEntry] = []
    part: str | None = None
    pending: TocEntry | None = None
    for md in tables:
        for row in markdown_table_rows(md):
            cells = [c for c in row if c]
            if not cells:
                continue
            pages = [int(x) for x in _INT.findall(cells[-1])] if len(cells) > 1 else []
            item_cell = next((c for c in cells if _TOC_ITEM.match(c)), None)
            part_m = _TOC_PART.search(" ".join(c for c in cells if c is not item_cell))
            if part_m or (item_cell and (part_m := _TOC_PART.search(item_cell))):
                part = f"Part {part_m.group(1).upper()}"
            if item_cell is not None:
                im = _TOC_ITEM.match(item_cell)
                assert im is not None
                rest = [c for c in cells if c is not item_cell and not _INT.fullmatch(c)]
                title = _title(im.group(2) or (rest[0] if rest else ""))
                pending = TocEntry(part, im.group(1).upper(), title, pages[0] if pages else -1)
                entries.append(pending)
            elif pending is not None and pending.printed_page < 0 and pages:
                pending.printed_page = pages[0]
    return [e for e in entries if e.printed_page > 0]


def _toc_anchor_positions(
    items: list[RawItem],
    heads: list[_Heading | None],
    form: str,
    n_pages: int,
    pages: dict[int, int],
) -> dict[int, list[_Heading]]:
    """Index in `items` -> anchors to insert before it, for Items the headings missed."""
    if form not in ("10-K", "10-Q"):
        return {}
    front = max(3, int(0.15 * n_pages))
    toc = [
        it for it in items if it.kind == "table" and it.page <= front and "item" in it.text.lower()
    ]
    entries = parse_toc([it.text for it in toc])
    if len(entries) < 3:
        return {}
    found: set[tuple[str | None, str]] = set()
    part: str | None = None
    for h in heads:
        if h is not None and h.level == "part":
            part = h.label.split(".")[0]
        elif h is not None and h.level == "item":
            m = re.match(r"Item (\d{1,2}[A-D]?)\.", h.label)
            if m:
                # 10-Q items before any Part heading belong to Part I.
                found.add(((part or "Part I") if form == "10-Q" else None, m.group(1)))
    toc_page = max(it.page for it in toc)
    by_printed: dict[int, int] = {}
    for page_no in sorted(pages):
        if page_no > toc_page:
            by_printed.setdefault(pages[page_no], page_no)

    def physical(printed: int) -> int | None:
        if printed in by_printed:
            return by_printed[printed]
        lower = [q for q in by_printed if q < printed and printed - q <= 10]
        if not lower:
            return None
        q = max(lower)
        return by_printed[q] + (printed - q)

    first_index: dict[int, int] = {}
    for i, it in enumerate(items):
        first_index.setdefault(it.page, i)
    out: dict[int, list[_Heading]] = {}
    for e in entries:
        key = (e.part if form == "10-Q" else None, e.number)
        if key in found:
            continue
        phys = physical(e.printed_page)
        if phys is None or phys <= toc_page:
            continue
        start = next((first_index[p] for p in range(phys, n_pages + 1) if p in first_index), None)
        if start is None:
            continue
        label = f"Item {e.number}. {e.title}" if e.title else f"Item {e.number}."
        out.setdefault(start, []).append(_Heading("anchor", label, part=e.part))
        found.add(key)
    return out


# ---------------------------------------------------------------- headings


def drop_headers_footers(items: list[RawItem], n_pages: int) -> list[RawItem]:
    """Remove page numbers and text repeated at the top or bottom of many pages.

    A repeated running header that names a well-known section ("Management's
    discussion and analysis") is kept as an `anchor` item instead of being dropped.
    """
    repeated = _repeated_edge_keys(items, n_pages)
    out = []
    for it in items:
        if it.kind == "para" and it.edge:
            if is_page_number(it.text):
                continue
            if len(it.text) <= 100 and _DIGITS.sub("#", it.text.lower()) in repeated:
                if anchor_key(it.text):
                    out.append(RawItem("anchor", it.text, it.page))
                continue
        out.append(it)
    return out


def _classify(items: list[RawItem], form: str, repair: bool = False) -> list[_Heading | None]:
    """Decide which paragraphs are headings. Item titles split over two lines are joined."""
    heads: list[_Heading | None] = [None] * len(items)
    skip: set[int] = set()
    for i, it in enumerate(items):
        if it.kind == "anchor":
            key = anchor_key(it.text)
            heads[i] = _anchor_heading(key, form) if key else None
            continue
        if it.kind != "para" or i in skip:
            continue
        text = repair_spacing(it.text) if repair and len(it.text) <= MAX_ITEM_LEN else it.text
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
                        title = _title(repair_spacing(cand) if repair else cand)
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
        if it.kind == "anchor" or (h is not None and h.level == "merged"):
            return True
        if h is not None or it.kind != "para":
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


class _Tracker:
    """The current Part / Item / sub-heading and the section path they make."""

    def __init__(self, form: str) -> None:
        self.form = form
        self.part: str | None = None
        self.item: str | None = None
        self.sub: str | None = None
        self.anchored = False  # the current item came from an anchor, not a real heading

    def path(self) -> str:
        parts: list[str] = []
        if self.part and (self.form == "10-Q" or self.item is None):
            parts.append(self.part)
        if self.item:
            parts.append(self.item)
        if not parts:
            parts.append(COVER)  # cover-page headings nest under Cover
        if self.sub:
            parts.append(self.sub)
        return " > ".join(parts)

    def anchor_allowed(self) -> bool:
        if self.item is None or self.anchored:
            return True
        n = re.match(r"Item (\d{1,2})", self.item)
        return self.form == "10-K" and n is not None and int(n.group(1)) >= 15

    def anchor(self, h: _Heading, forced: bool) -> bool:
        """Switch to an anchored Item. Returns False when not allowed or not a change."""
        if not forced and not self.anchor_allowed():
            return False
        if h.label == self.item and (h.part is None or h.part == self.part):
            return False
        if h.part is not None:
            self.part = h.part
        self.item, self.sub, self.anchored = h.label, None, True
        return True

    def heading(self, h: _Heading) -> bool:
        """Apply a heading. Returns False for a repeat of the current Part (a page header)."""
        if h.level == "part" and h.label.split(".")[0] == self.part:
            return False
        if h.level == "part":
            self.part, self.item, self.sub, self.anchored = h.label.split(".")[0], None, None, False
        elif h.level == "item":
            self.item, self.sub, self.anchored = h.label, None, False
        else:
            self.sub = h.label
        return True


def build_blocks(
    items: list[RawItem], form: str, n_pages: int, *, repair_spacing_headings: bool = False
) -> list[Block]:
    """Assign headings and section paths and return Blocks in reading order."""
    pages = printed_pages(items, _repeated_edge_keys(items, n_pages))
    items = drop_headers_footers(items, n_pages)
    heads = _classify(items, form, repair=repair_spacing_headings)
    _suppress_toc(items, heads)
    toc_anchors = _toc_anchor_positions(items, heads, form, n_pages, pages)
    state = _Tracker(form)
    blocks: list[Block] = []

    def emit_heading(label: str, page: int) -> None:
        blocks.append(Block(kind="heading", text=label, section=state.path(), page=page))

    for i, (it, h) in enumerate(zip(items, heads, strict=True)):
        for a in toc_anchors.get(i, []):
            if state.anchor(a, forced=True):
                emit_heading(a.label, it.page)
        if it.kind == "anchor":
            if h is not None and state.anchor(h, forced=False):
                emit_heading(h.label, it.page)
            continue
        if h is not None and h.level == "merged":
            continue
        if h is not None and h.level == "sub" and (key := anchor_key(h.label)):
            ah = _anchor_heading(key, form)
            if ah is not None and state.anchor(ah, forced=False):
                emit_heading(ah.label, it.page)
        if h is not None:
            if state.heading(h):
                emit_heading(h.label, it.page)
            continue
        if it.kind == "table":
            blocks.append(
                Block(
                    kind="table",
                    text=it.text,
                    section=state.path(),
                    page=it.page,
                    table_id=it.table_id,
                )
            )
        elif it.text:
            blocks.append(Block(kind="text", text=it.text, section=state.path(), page=it.page))
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


def item_key(section: str) -> tuple[str | None, str] | None:
    """`Part I > Item 2. MD&A > Liquidity` -> ("Part I", "2"); Part is None in a 10-K."""
    number = item_number(section)
    if number is None:
        return None
    part = next((p for p in section.split(" > ") if PART_RE.match(p)), None)
    return part, number
