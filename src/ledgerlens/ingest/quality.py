"""Data quality checks over parsed filings and chunks.

Counts are written to `DATA_DIR/reports/quality.json` (full detail, with examples)
and `quality.md` (summary tables). Checks:

- empty sections: an Item section with no text or table under it (Part and cover
  headings are structural and not counted),
  and sub-sections whose heading is followed directly by another heading;
- broken tables: ragged rows, all-empty columns, header-only tables;
- duplicate chunks: exact (hash of the body text) and near (hash of lowercased
  alphanumerics only), per strategy across the corpus;
- chunk size outside bounds (under `MIN_TOKENS` or over `MAX_TOKENS`);
- 10-K filings with no Item 7 or Item 8 heading, 10-Qs with no Part I Item 1 or
  Part I Item 2 (Part II has its own Items 1 and 2, which do not count);
- 10-K/10-Q filings with more than `MAX_OUTSIDE_ITEM_SHARE` of their content blocks
  outside any Item section (a sign the section paths failed);
- table cells that hold more than one number (merged value columns).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ledgerlens.ingest.chunk import approx_tokens, body_text
from ledgerlens.ingest.sections import item_key, top_level_section
from ledgerlens.ingest.tables import markdown_table_rows, packed_cell
from ledgerlens.schemas import Chunk, ParsedFiling

MIN_TOKENS = 20
MAX_TOKENS = 1200
MAX_EXAMPLES = 10
MAX_OUTSIDE_ITEM_SHARE = 0.5
REQUIRED_ITEMS: dict[str, tuple[tuple[str | None, str], ...]] = {
    "10-K": ((None, "7"), (None, "8")),
    "10-Q": (("Part I", "1"), ("Part I", "2")),
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def table_problems(markdown: str) -> list[str]:
    """Return the problems found in one Markdown table (empty list when fine)."""
    rows = markdown_table_rows(markdown)
    if not rows:
        return ["unparseable"]
    problems = []
    if len({len(r) for r in rows}) > 1:
        problems.append("ragged_rows")
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    if any(not any(r[j] for r in padded) for j in range(width)):
        problems.append("empty_column")
    if len(rows) < 2:
        problems.append("header_only")
    return problems


def empty_sections(parsed: ParsedFiling) -> tuple[list[str], list[str]]:
    """(empty Item sections, empty sub-sections) for one filing."""
    content: Counter[str] = Counter()
    order: list[str] = []
    empty_subs: list[str] = []
    blocks = parsed.blocks
    for i, b in enumerate(blocks):
        top = top_level_section(b.section)
        if top not in content:
            order.append(top)
            content[top] += 0
        if b.kind != "heading":
            content[top] += 1
        elif " > " in b.section:
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if nxt is None or nxt.kind == "heading":
                empty_subs.append(b.section)
    return [s for s in order if content[s] == 0 and s.startswith("Item ")], empty_subs


def missing_items(parsed: ParsedFiling) -> list[str]:
    """Required Items with no heading. A 10-Q needs them under Part I."""
    required = REQUIRED_ITEMS.get(parsed.filing.form)
    if not required:
        return []
    found = {item_key(b.section) for b in parsed.blocks if b.kind == "heading"}
    return [
        f"{part} Item {n}" if part else f"Item {n}"
        for part, n in required
        if (part, n) not in found
    ]


def outside_item_share(parsed: ParsedFiling) -> float | None:
    """Share of content blocks not under any Item (10-K and 10-Q only)."""
    if parsed.filing.form not in REQUIRED_ITEMS:
        return None
    content = [b for b in parsed.blocks if b.kind != "heading"]
    if not content:
        return None
    outside = sum(1 for b in content if item_key(b.section) is None)
    return round(outside / len(content), 3)


def packed_cells(markdown: str) -> int:
    """Body cells holding two or more numbers side by side."""
    return sum(packed_cell(c) for row in markdown_table_rows(markdown)[1:] for c in row)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dupes(chunks: list[Chunk], key: Any) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        body = body_text(c).strip()
        if body:
            groups[key(body)].append(c.chunk_id)
    dup_groups = [ids for ids in groups.values() if len(ids) > 1]
    cross = [ids for ids in dup_groups if len({i.rsplit(":", 2)[0] for i in ids}) > 1]
    return {
        "groups": len(dup_groups),
        "extra_chunks": sum(len(g) - 1 for g in dup_groups),
        "cross_filing_groups": len(cross),
        "examples": dup_groups[:MAX_EXAMPLES],
    }


def check(parsed: list[ParsedFiling], chunks: dict[str, list[Chunk]]) -> dict[str, Any]:
    """Run every check and return a JSON-serialisable report."""
    filings: list[dict[str, Any]] = []
    table_issue_counts: Counter[str] = Counter()
    table_examples: list[dict[str, Any]] = []
    n_tables = 0
    n_packed = 0
    for p in parsed:
        empty_top, empty_sub = empty_sections(p)
        missing = missing_items(p)
        issues: Counter[str] = Counter()
        for b in p.blocks:
            if b.kind != "table":
                continue
            n_tables += 1
            n_packed += packed_cells(b.text)
            for prob in table_problems(b.text):
                issues[prob] += 1
                table_issue_counts[prob] += 1
                if len(table_examples) < MAX_EXAMPLES:
                    table_examples.append({"table_id": b.table_id, "problem": prob})
        kinds = Counter(b.kind for b in p.blocks)
        filings.append(
            {
                "filing_id": p.filing.filing_id,
                "form": p.filing.form,
                "source_format": p.filing.source_format,
                "blocks": len(p.blocks),
                "headings": kinds["heading"],
                "tables": kinds["table"],
                "max_page": max((b.page or 0 for b in p.blocks), default=0),
                "empty_sections": empty_top,
                "empty_subsections": len(empty_sub),
                "missing_items": missing,
                "outside_item_share": outside_item_share(p),
                "table_issues": dict(issues),
            }
        )

    strategies: dict[str, Any] = {}
    for strategy, items in sorted(chunks.items()):
        sizes = [approx_tokens(c.text) for c in items]
        under = [c.chunk_id for c, s in zip(items, sizes, strict=True) if s < MIN_TOKENS]
        over = [c.chunk_id for c, s in zip(items, sizes, strict=True) if s > MAX_TOKENS]
        sorted_sizes = sorted(sizes)
        strategies[strategy] = {
            "chunks": len(items),
            "with_table": sum(c.has_table for c in items),
            "tokens_mean": round(sum(sizes) / len(sizes), 1) if sizes else 0,
            "tokens_p50": sorted_sizes[len(sizes) // 2] if sizes else 0,
            "tokens_max": max(sizes, default=0),
            "under_min": len(under),
            "over_max": len(over),
            "under_examples": under[:MAX_EXAMPLES],
            "over_examples": over[:MAX_EXAMPLES],
            "exact_duplicates": _dupes(items, _hash),
            "near_duplicates": _dupes(items, lambda t: _hash(_NON_ALNUM.sub("", t.lower()))),
        }

    totals = {
        "filings": len(parsed),
        "blocks": sum(f["blocks"] for f in filings),
        "tables": n_tables,
        "filings_with_empty_sections": sum(bool(f["empty_sections"]) for f in filings),
        "empty_sections": sum(len(f["empty_sections"]) for f in filings),
        "empty_subsections": sum(f["empty_subsections"] for f in filings),
        "broken_tables": dict(table_issue_counts),
        "broken_tables_total": sum(table_issue_counts.values()),
        "multi_number_cells": n_packed,
        "filings_mostly_outside_items": [
            {"filing_id": f["filing_id"], "share": f["outside_item_share"]}
            for f in filings
            if (f["outside_item_share"] or 0) > MAX_OUTSIDE_ITEM_SHARE
        ],
        "filings_missing_items": [
            {"filing_id": f["filing_id"], "missing": f["missing_items"]}
            for f in filings
            if f["missing_items"]
        ],
    }
    return {
        "bounds": {"min_tokens": MIN_TOKENS, "max_tokens": MAX_TOKENS},
        "totals": totals,
        "strategies": strategies,
        "table_examples": table_examples,
        "filings": filings,
    }


def to_markdown(report: dict[str, Any]) -> str:
    t = report["totals"]
    b = report["bounds"]
    lines = [
        "# Data quality report",
        "",
        "Generated by `ledgerlens ingest run` (`src/ledgerlens/ingest/quality.py`).",
        "",
        "## Corpus",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Filings | {t['filings']} |",
        f"| Blocks | {t['blocks']} |",
        f"| Tables | {t['tables']} |",
        f"| Empty Item sections | {t['empty_sections']} "
        f"(in {t['filings_with_empty_sections']} filings) |",
        f"| Empty sub-sections | {t['empty_subsections']} |",
        f"| Broken tables | {t['broken_tables_total']} |",
        f"| Table cells holding more than one number | {t['multi_number_cells']} |",
        f"| 10-K/10-Q filings with over {MAX_OUTSIDE_ITEM_SHARE:.0%} of blocks outside any Item "
        f"| {len(t['filings_mostly_outside_items'])} |",
    ]
    for prob, n in sorted(t["broken_tables"].items()):
        lines.append(f"| Broken tables: {prob} | {n} |")
    lines.append(
        "| Filings missing Item 7/8 (10-K) or Part I Item 1/2 (10-Q) "
        f"| {len(t['filings_missing_items'])} |"
    )
    lines += [
        "",
        "## Chunks",
        "",
        f"Size bounds: {b['min_tokens']} to {b['max_tokens']} approximate tokens (words x 1.3).",
        "",
        "| Strategy | Chunks | With table | Mean tokens | Median | Max | Under min | Over max "
        "| Exact dup groups | Near dup groups | Cross-filing near dups |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in report["strategies"].items():
        lines.append(
            f"| {name} | {s['chunks']} | {s['with_table']} | {s['tokens_mean']} | "
            f"{s['tokens_p50']} | {s['tokens_max']} | {s['under_min']} | {s['over_max']} | "
            f"{s['exact_duplicates']['groups']} | {s['near_duplicates']['groups']} | "
            f"{s['near_duplicates']['cross_filing_groups']} |"
        )
    if t["filings_missing_items"]:
        lines += ["", "## Filings missing required items", ""]
        lines += [
            f"- {m['filing_id']}: {', '.join(m['missing'])}" for m in t["filings_missing_items"]
        ]
    if t["filings_mostly_outside_items"]:
        lines += ["", "## Filings with most blocks outside any Item", ""]
        lines += [
            f"- {m['filing_id']}: {m['share']:.0%}" for m in t["filings_mostly_outside_items"]
        ]
    lines += [
        "",
        "## Per filing",
        "",
        "| Filing | Format | Blocks | Headings | Tables | Pages | Outside Items "
        "| Empty sections | Table issues |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in report["filings"]:
        issues = sum(f["table_issues"].values())
        share = f.get("outside_item_share")
        outside = "n/a" if share is None else f"{share:.0%}"
        lines.append(
            f"| {f['filing_id']} | {f['source_format']} | {f['blocks']} | {f['headings']} | "
            f"{f['tables']} | {f['max_page']} | {outside} | {len(f['empty_sections'])} | "
            f"{issues} |"
        )
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "quality.json"
    md_path = reports_dir / "quality.md"
    _write_atomic(json_path, json.dumps(report, indent=2) + "\n")
    _write_atomic(md_path, to_markdown(report))
    return json_path, md_path
