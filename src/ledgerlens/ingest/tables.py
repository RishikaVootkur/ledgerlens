"""Turn a grid of table cells into a clean GitHub Markdown table.

EDGAR tables are laid out for print: a `$` sits in its own cell left of the number,
the closing `)` of a negative and the `%` sign sit in their own cell to the right,
and narrow empty columns act as spacers. Headers often span columns that the body
splits. This module undoes all of that so that each logical column becomes one
Markdown column. HTML and PDF parsing share it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_WS = re.compile(r"\s+")
_PREFIX_CELLS = {"$", "€", "£", "¥", "$(", "($", "(", "US$"}
_SUFFIX_CELLS = {")", "%", ")%", "%)", "pts", "bps", "x"}
_NUMERIC = re.compile(r"^[\$€£¥(]*-?[\d.,]+\)?%?$|^[—–-]+$")
_OPEN_PAREN = re.compile(r"\(\s+(?=[\d$.])")
_CLOSE_PAREN = re.compile(r"(?<=[\d%])\s+\)")
_CURRENCY_GAP = re.compile(r"([$€£¥])\s+(?=[\d(])")
_YEAR = re.compile(r"^(19|20)\d{2}$")
_BULLETS = {"•", "●", "◦", "▪", "■", "-", "–", "—", "·", "o"}


def clean_cell(text: str | None) -> str:
    """Collapse whitespace (including non-breaking spaces), tighten `( 321 )` and `$ 5`."""
    if not text:
        return ""
    text = _WS.sub(" ", text.replace("\xa0", " ").replace("\u200b", "")).strip()
    text = _OPEN_PAREN.sub("(", text)
    text = _CLOSE_PAREN.sub(")", text)
    return _CURRENCY_GAP.sub(r"\1", text)


def is_numeric_cell(text: str) -> bool:
    return bool(_NUMERIC.match(text.replace(" ", "")))


def _merge_symbols(row: list[str]) -> list[str]:
    """Glue `$` onto the number to its right and `)`/`%` onto the value to its left."""
    out = list(row)
    n = len(out)
    for j in range(n):
        cell = out[j]
        if cell in _PREFIX_CELLS:
            k = next((k for k in range(j + 1, n) if out[k]), None)
            if k is not None and (is_numeric_cell(out[k]) or out[k][:1].isdigit()):
                out[k] = cell + out[k]
                out[j] = ""
        elif cell in _SUFFIX_CELLS:
            k = next((k for k in range(j - 1, -1, -1) if out[k]), None)
            if k is not None and k > 0 and (is_numeric_cell(out[k]) or out[k][-1:].isdigit()):
                out[k] = out[k] + cell
                out[j] = ""
    return out


def _drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    width = len(rows[0])
    keep = [j for j in range(width) if any(r[j] for r in rows)]
    return [[r[j] for j in keep] for r in rows]


def _merge_complementary_columns(rows: list[list[str]]) -> list[list[str]]:
    """Merge neighbour columns that never both hold a value in the same row.

    Spanned header cells and their body values land in adjacent grid columns; this
    folds them back together. The first (label) column is never merged.
    """
    changed = True
    while changed:
        changed = False
        width = len(rows[0])
        for j in range(1, width - 1):
            if not any(r[j] and r[j + 1] for r in rows):
                for r in rows:
                    r[j] = r[j] or r[j + 1]
                    del r[j + 1]
                changed = True
                break
    return rows


def _is_data_row(row: list[str]) -> bool:
    """A labelled row with at least one number that is not a bare year."""
    return bool(row[0]) and any(is_numeric_cell(c) and not _YEAR.match(c) for c in row[1:] if c)


def _is_label_row(row: list[str]) -> bool:
    return bool(row[0]) and not any(row[1:])


def _escape(cell: str) -> str:
    return cell.replace("|", "\\|")


def normalize_rows(raw_rows: Sequence[Sequence[str | None]]) -> list[list[str]]:
    """Clean cells, pad ragged rows, merge symbols and drop spacer columns and empty rows."""
    rows = [[clean_cell(c) for c in r] for r in raw_rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    rows = [_merge_symbols(r) for r in rows]
    rows = _drop_empty_columns(rows)
    if rows and len(rows[0]) > 2:
        rows = _merge_complementary_columns(rows)
    return rows


_AMOUNT = re.compile(r"^\(?[$€£¥]?\(?-?\d[\d,]*(\.\d+)?\)?%?\)?$|^[—–-]$")


def number_tokens(cell: str) -> list[str] | None:
    """Split a cell holding only numbers ("$4,915 $4,891") into tokens, else None.

    A lone currency sign is glued to the number after it. An em dash counts as a
    missing value; an en dash or hyphen between numbers marks a range (None).
    """
    out: list[str] = []
    pending = ""
    for tok in cell.split():
        if tok in _PREFIX_CELLS:
            pending += tok
            continue
        tok = pending + tok
        pending = ""
        if not _AMOUNT.match(tok):
            return None
        out.append(tok)
    if pending:
        return None
    # "2025 – 2062" is a range, not two values: en dashes and hyphens between numbers.
    if any(t in ("–", "-") for t in out[1:-1]):
        return None
    return out


def packed_cell(cell: str) -> bool:
    """True for a cell that holds two or more numbers side by side."""
    toks = number_tokens(cell)
    return toks is not None and len(toks) > 1


def split_packed_columns(rows: list[list[str]]) -> list[list[str]]:
    """Split columns where PDF extraction put several value columns into one cell.

    A column is split into k columns only when every non-empty cell in it either
    holds exactly k numbers (k >= 2, at least two such cells) or is plain text
    (a spanning label, kept in the first new column). Anything less regular is
    left alone, because the year-to-value mapping would be a guess.
    """
    j = 1
    while rows and j < len(rows[0]):
        counts = []
        regular = True
        for r in rows:
            cell = r[j]
            if not cell:
                continue
            toks = number_tokens(cell)
            if toks is None:
                continue  # text label
            if len(toks) < 2:
                regular = False
                break
            counts.append(len(toks))
        k = counts[0] if counts else 0
        if regular and len(counts) >= 2 and all(c == k for c in counts):
            for r in rows:
                cell = r[j]
                toks = number_tokens(cell) if cell else None
                parts = toks if toks is not None else [cell] + [""] * (k - 1)
                r[j : j + 1] = parts
            j += k
        else:
            j += 1
    return rows


def rows_to_text(rows: list[list[str]]) -> list[str]:
    """Flatten a layout table into one line of text per row."""
    lines = [" ".join(c for c in r if c) for r in rows]
    return [ln for ln in lines if ln]


def rows_to_markdown(raw_rows: Sequence[Sequence[str | None]]) -> str | None:
    """Return a GitHub Markdown table, or None when the grid is really layout, not data.

    Layout tables (one column, one row, or a bullet column next to text) should be
    treated as prose by the caller; use :func:`rows_to_text` for that.
    """
    rows = split_packed_columns(normalize_rows(raw_rows))
    if len(rows) < 2 or len(rows[0]) < 2:
        return None
    if all(r[0] in _BULLETS for r in rows if r[0]):
        return None
    # Rows before the first data row are header rows, folded into one. Row labels
    # right above the data ("Net sales:") belong to the body; a table that starts
    # straight with data gets a blank header row.
    width = len(rows[0])
    first_data = next((i for i, r in enumerate(rows) if _is_data_row(r)), None)
    if first_data is None:
        header_rows, body = rows[:1], rows[1:]
    else:
        n_header = min(first_data, 4)
        header_rows, body = rows[:n_header], rows[n_header:]
        while header_rows and _is_label_row(header_rows[-1]):
            body.insert(0, header_rows.pop())
    if not body:
        return None
    header = [" ".join(h[j] for h in header_rows if h[j]) for j in range(width)]
    lines = [
        "| " + " | ".join(_escape(c) for c in header) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    lines += ["| " + " | ".join(_escape(c) for c in r) + " |" for r in body]
    return "\n".join(lines)


def markdown_table_rows(markdown: str) -> list[list[str]]:
    """Parse a Markdown table produced by :func:`rows_to_markdown` back into cells.

    The separator line is dropped; the header is row 0.
    """
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.fullmatch(r"\|(\s*:?-{3,}:?\s*\|)+", line):
            continue
        inner = line[1:-1] if line.endswith("|") and not line.endswith("\\|") else line[1:]
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", inner)]
        rows.append(cells)
    return rows
