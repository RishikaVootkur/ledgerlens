import pytest

from ledgerlens.ingest.chunk import (
    approx_tokens,
    body_text,
    chunk_fixed,
    chunk_section,
    context_header,
    split_table,
)
from ledgerlens.schemas import Block, FilingRef, ParsedFiling

MDA = "Item 7. MD&A"
LIQ = "Item 7. MD&A > Liquidity"


def _words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def _table(rows: int) -> str:
    lines = ["|  | 2025 | 2024 |", "|---|---|---|"]
    lines += [f"| Line item {i} | {i},000 | {i},500 |" for i in range(rows)]
    return "\n".join(lines)


@pytest.fixture
def parsed(html_ref: FilingRef) -> ParsedFiling:
    blocks = [
        Block(kind="heading", text="Item 7. MD&A", section=MDA, page=30),
        Block(kind="text", text=_words("intro", 300), section=MDA, page=30),
        Block(kind="text", text=_words("more", 300), section=MDA, page=31),
        Block(kind="heading", text="Liquidity", section=LIQ, page=32),
        Block(kind="text", text=_words("cash", 50), section=LIQ, page=32),
        Block(kind="table", text=_table(5), section=LIQ, page=32, table_id="T:table:0001"),
        Block(kind="text", text=_words("debt", 40), section=LIQ, page=33),
    ]
    return ParsedFiling(filing=html_ref, blocks=blocks)


def test_fixed_ids_size_and_overlap(parsed):
    chunks = chunk_fixed(parsed)
    assert [c.chunk_id for c in chunks] == [
        f"SMPL-10K-2025FY:fixed:{i:05d}" for i in range(len(chunks))
    ]
    assert all(approx_tokens(c.text) <= 512 for c in chunks)
    assert approx_tokens(chunks[0].text) >= 500
    # The last ~49 words of one chunk open the next one.
    tail = chunks[0].text.split()[-49:]
    assert chunks[1].text.split()[:49] == tail
    assert chunk_fixed(parsed) == chunks  # deterministic


def test_fixed_ignores_structure(parsed):
    chunks = chunk_fixed(parsed)
    assert any(MDA in c.section for c in chunks)
    crossing = [c for c in chunks if "more299" in c.text and "cash0" in c.text]
    assert crossing, "fixed windows run across section boundaries"


def test_section_never_crosses_sections(parsed):
    chunks = chunk_section(parsed)
    for c in chunks:
        body = body_text(c)
        if c.section == MDA:
            assert "cash" not in body and "debt" not in body
        else:
            assert "intro" not in body and "more" not in body


def test_section_keeps_table_whole_and_alone(parsed):
    chunks = chunk_section(parsed)
    table_chunks = [c for c in chunks if c.has_table]
    assert len(table_chunks) == 1
    t = table_chunks[0]
    assert t.table_ids == ["T:table:0001"]
    assert body_text(t) == _table(5)
    assert t.page == 32 and t.page_end == 32


def test_section_context_header_and_pages(parsed):
    chunks = chunk_section(parsed)
    first = chunks[0]
    assert first.text.startswith("[SMPL 10-K FY2025 | Item 7. MD&A | p.30]\n")
    assert all(c.chunk_id == f"SMPL-10K-2025FY:section:{i:05d}" for i, c in enumerate(chunks))
    # 600 words of MD&A cannot fit in one ~512 token chunk.
    assert len([c for c in chunks if c.section == MDA]) >= 2
    # The short tail after the table merges into nothing before a table: it stands alone.
    assert body_text(chunks[-1]).startswith("debt0")
    assert chunk_section(parsed) == chunks


def test_long_table_split_repeats_header():
    parts = split_table(_table(200), max_tokens=300)
    assert len(parts) > 1
    for p in parts:
        lines = p.splitlines()
        assert lines[:2] == ["|  | 2025 | 2024 |", "|---|---|---|"]
        assert approx_tokens(p) <= 300
    rows = [ln for p in parts for ln in p.splitlines()[2:]]
    assert len(rows) == 200


def test_context_header_for_quarter(html_ref):
    q = html_ref.model_copy(update={"form": "10-Q", "fiscal_period": "Q2", "fiscal_year": 2026})
    assert context_header(q, "Part I > Item 2. MD&A", 5, 6) == (
        "[SMPL 10-Q Q2 FY2026 | Part I > Item 2. MD&A | pp.5-6]"
    )
    assert context_header(q, "Cover", None, None) == "[SMPL 10-Q Q2 FY2026 | Cover]"
