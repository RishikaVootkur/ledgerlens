"""Two chunking strategies over a ParsedFiling.

Token counts are approximated as words * 1.3 (no tokenizer dependency).

`fixed`: a sliding window of about 512 tokens with 64 tokens of overlap over the
whole document text, ignoring headings and tables. The baseline.

`section`: never crosses a section boundary; packs paragraphs up to about 512
tokens; puts every table in its own chunk (very long tables are split by rows and
each part repeats the header); prefixes each chunk with a one-line context header
such as `[AAPL 10-K FY2025 | Item 7. MD&A > Liquidity | p.32]`.

Chunk ids are `{filing_id}:{strategy}:{seq:05d}`, so the same input always yields
the same ids.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ledgerlens.schemas import Block, Chunk, ChunkStrategy, FilingRef, ParsedFiling, make_chunk_id

TOKENS_PER_WORD = 1.3
TARGET_TOKENS = 512
OVERLAP_TOKENS = 64
MAX_TABLE_TOKENS = 1024
MIN_TAIL_TOKENS = 64
CHUNKER_VERSION = "1"

_TOKEN_RE = re.compile(r"\S+\s*")


def approx_tokens(text: str) -> int:
    return round(len(text.split()) * TOKENS_PER_WORD)


def _words_for(tokens: int) -> int:
    return max(1, round(tokens / TOKENS_PER_WORD))


def period_label(filing: FilingRef) -> str:
    if filing.fiscal_period == "FY":
        return f"FY{filing.fiscal_year}"
    return f"{filing.fiscal_period} FY{filing.fiscal_year}"


def page_label(page: int | None, page_end: int | None) -> str | None:
    if page is None:
        return None
    if page_end is None or page_end == page:
        return f"p.{page}"
    return f"pp.{page}-{page_end}"


def context_header(filing: FilingRef, section: str, page: int | None, page_end: int | None) -> str:
    parts = [f"{filing.ticker} {filing.form} {period_label(filing)}", section]
    if (pl := page_label(page, page_end)) is not None:
        parts.append(pl)
    return "[" + " | ".join(parts) + "]"


def _make_chunk(
    filing: FilingRef,
    strategy: ChunkStrategy,
    seq: int,
    section: str,
    text: str,
    blocks: list[Block],
) -> Chunk:
    pages = [b.page for b in blocks if b.page is not None]
    table_ids = [b.table_id for b in blocks if b.kind == "table" and b.table_id]
    return Chunk(
        chunk_id=make_chunk_id(filing.filing_id, strategy, seq),
        filing_id=filing.filing_id,
        ticker=filing.ticker,
        company=filing.company,
        form=filing.form,
        fiscal_year=filing.fiscal_year,
        fiscal_period=filing.fiscal_period,
        doc_name=filing.doc_name,
        strategy=strategy,
        section=section,
        page=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        text=text,
        has_table=bool(table_ids),
        table_ids=list(dict.fromkeys(table_ids)),
    )


# ---------------------------------------------------------------- fixed


def chunk_fixed(
    parsed: ParsedFiling,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Sliding word window over the whole document; structure is ignored."""
    size = _words_for(target_tokens)
    overlap = min(_words_for(overlap_tokens), size - 1)
    step = size - overlap
    words: list[str] = []
    owner: list[int] = []
    for i, block in enumerate(parsed.blocks):
        toks = _TOKEN_RE.findall(block.text)
        if not toks:
            continue
        toks[-1] = toks[-1].rstrip() + "\n\n"
        words.extend(toks)
        owner.extend([i] * len(toks))
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        covered = sorted(set(owner[start:end]))
        blocks = [parsed.blocks[i] for i in covered]
        text = "".join(words[start:end]).strip()
        chunks.append(
            _make_chunk(parsed.filing, "fixed", len(chunks), blocks[0].section, text, blocks)
        )
        if end == len(words):
            break
        start += step
    return chunks


# ---------------------------------------------------------------- section


@dataclass
class _Piece:
    """Body text of one future chunk plus the blocks it came from."""

    text: str
    blocks: list[Block] = field(default_factory=list)
    is_table: bool = False


def split_table(markdown: str, max_tokens: int = MAX_TABLE_TOKENS) -> list[str]:
    """Split a Markdown table by rows; every part repeats the header and separator."""
    if approx_tokens(markdown) <= max_tokens:
        return [markdown]
    lines = markdown.splitlines()
    head, rows = lines[:2], lines[2:]
    head_tokens = approx_tokens("\n".join(head))
    parts: list[str] = []
    current: list[str] = []
    current_tokens = head_tokens
    for row in rows:
        t = approx_tokens(row)
        if current and current_tokens + t > max_tokens:
            parts.append("\n".join(head + current))
            current, current_tokens = [], head_tokens
        current.append(row)
        current_tokens += t
    if current:
        parts.append("\n".join(head + current))
    return parts


def _split_long_text(text: str, target_tokens: int) -> list[str]:
    words = text.split()
    size = _words_for(target_tokens)
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def _section_pieces(blocks: list[Block], target_tokens: int, max_table_tokens: int) -> list[_Piece]:
    pieces: list[_Piece] = []
    buf: list[Block] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        text = "\n\n".join(b.text for b in buf)
        prev = pieces[-1] if pieces else None
        if (
            buf_tokens < MIN_TAIL_TOKENS
            and prev is not None
            and not prev.is_table
            and approx_tokens(prev.text) + buf_tokens <= int(target_tokens * 1.25)
        ):
            prev.text += "\n\n" + text
            prev.blocks.extend(buf)
        else:
            pieces.append(_Piece(text, list(buf)))
        buf, buf_tokens = [], 0

    for block in blocks:
        if block.kind == "heading":
            continue
        tokens = approx_tokens(block.text)
        if block.kind == "table":
            flush()
            for part in split_table(block.text, max_table_tokens):
                pieces.append(_Piece(part, [block], is_table=True))
            continue
        if tokens > target_tokens:
            flush()
            for part in _split_long_text(block.text, target_tokens):
                pieces.append(_Piece(part, [block]))
            continue
        if buf and buf_tokens + tokens > target_tokens:
            flush()
        buf.append(block)
        buf_tokens += tokens
    flush()
    return pieces


def _groups(blocks: list[Block]) -> list[tuple[str, list[Block]]]:
    """Consecutive blocks sharing a section; a heading always opens a new group."""
    groups: list[tuple[str, list[Block]]] = []
    for block in blocks:
        if not groups or block.kind == "heading" or block.section != groups[-1][0]:
            groups.append((block.section, [block]))
        else:
            groups[-1][1].append(block)
    return groups


def chunk_section(
    parsed: ParsedFiling,
    target_tokens: int = TARGET_TOKENS,
    max_table_tokens: int = MAX_TABLE_TOKENS,
) -> list[Chunk]:
    """Section-bounded, table-aware chunks with a context header line."""
    chunks: list[Chunk] = []
    for section, blocks in _groups(parsed.blocks):
        for piece in _section_pieces(blocks, target_tokens, max_table_tokens):
            pages = [b.page for b in piece.blocks if b.page is not None]
            header = context_header(
                parsed.filing,
                section,
                min(pages) if pages else None,
                max(pages) if pages else None,
            )
            chunks.append(
                _make_chunk(
                    parsed.filing,
                    "section",
                    len(chunks),
                    section,
                    f"{header}\n{piece.text}",
                    piece.blocks,
                )
            )
    return chunks


def body_text(chunk: Chunk) -> str:
    """Chunk text without the section strategy's context header line."""
    if chunk.strategy == "section" and chunk.text.startswith("["):
        return chunk.text.split("\n", 1)[1] if "\n" in chunk.text else ""
    return chunk.text


CHUNKERS: dict[str, Callable[[ParsedFiling], list[Chunk]]] = {
    "fixed": chunk_fixed,
    "section": chunk_section,
}
