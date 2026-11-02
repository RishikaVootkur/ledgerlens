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
- 10-K filings with no Item 7 or Item 8 heading, 10-Qs with no Item 1 or Item 2.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ledgerlens.ingest.chunk import approx_tokens, body_text
from ledgerlens.ingest.sections import item_number, top_level_section
from ledgerlens.ingest.tables import markdown_table_rows
from ledgerlens.schemas import Chunk, ParsedFiling

MIN_TOKENS = 20
MAX_TOKENS = 1200
MAX_EXAMPLES = 10
REQUIRED_ITEMS = {"10-K": ("7", "8"), "10-Q": ("1", "2")}

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
    required = REQUIRED_ITEMS.get(parsed.filing.form)
    if not required:
        return []
    found = {item_number(b.section) for b in parsed.blocks if b.kind == "heading"}
    return [f"Item {n}" for n in required if n not in found]


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
    for p in parsed:
        empty_top, empty_sub = empty_sections(p)
        missing = missing_items(p)
        issues: Counter[str] = Counter()
        for b in p.blocks:
            if b.kind != "table":
                continue
            n_tables += 1
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
    ]
    for prob, n in sorted(t["broken_tables"].items()):
        lines.append(f"| Broken tables: {prob} | {n} |")
    lines.append(
        f"| Filings missing Item 7/8 (10-K) or 1/2 (10-Q) | {len(t['filings_missing_items'])} |"
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
    lines += [
        "",
        "## Per filing",
        "",
        "| Filing | Format | Blocks | Headings | Tables | Pages | Empty sections | Table issues |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in report["filings"]:
        issues = sum(f["table_issues"].values())
        lines.append(
            f"| {f['filing_id']} | {f['source_format']} | {f['blocks']} | {f['headings']} | "
            f"{f['tables']} | {f['max_page']} | {len(f['empty_sections'])} | {issues} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "quality.json"
    md_path = reports_dir / "quality.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path
