"""Shared helpers for the dataset generators: paths, splits, JSONL IO and number formatting.

Split rule (documented in every datasheet): within each `question_type` of a dataset,
questions are ordered by the SHA-256 hex digest of their `qid`; the first
`round(0.4 * n)` are `dev`, the rest `test`. The assignment depends only on the qids,
so regenerating a dataset reproduces it exactly, and each stratum is 40/60 up to
rounding. The smoke set is the exception: it is all `dev` by design.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

from ledgerlens.config import get_settings
from ledgerlens.schemas import Question

DEV_FRACTION = 0.4


def datasets_dir() -> Path:
    return get_settings().eval_dir / "datasets"


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_splits(
    questions: Sequence[Question], dev_fraction: float = DEV_FRACTION
) -> list[Question]:
    """Return copies with `split` set by the stratified SHA-256 rule, in the input order."""
    strata: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        strata[q.question_type].append(q.qid)
    dev: set[str] = set()
    for qids in strata.values():
        ordered = sorted(qids, key=stable_hash)
        dev.update(ordered[: round(dev_fraction * len(ordered))])
    return [
        Question.model_validate({**q.model_dump(), "split": "dev" if q.qid in dev else "test"})
        for q in questions
    ]


def write_jsonl(path: Path, questions: Iterable[Question]) -> int:
    """Write one validated Question per line; returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(q.model_dump(mode="json"), ensure_ascii=False) for q in questions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def read_jsonl(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as fh:
        return [Question.model_validate_json(line) for line in fh if line.strip()]


def fmt_usd(value: float) -> str:
    """`$416,161,000,000 (about $416.16 billion)` style text for a USD amount."""
    sign = "-" if value < 0 else ""
    mag = abs(value)
    exact = f"{sign}${mag:,.0f}"
    for div, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if mag >= div:
            return f"{exact} (about {sign}${mag / div:,.2f} {word})"
    return exact


def fmt_date(d: date) -> str:
    """`September 27, 2025` for a date."""
    return f"{d:%B} {d.day}, {d.year}"
