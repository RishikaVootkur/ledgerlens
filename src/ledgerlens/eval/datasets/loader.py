"""Read the committed eval datasets (`eval/datasets/{name}.jsonl`) as `Question`s."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

from ledgerlens.eval.datasets.common import datasets_dir, read_jsonl
from ledgerlens.schemas import DatasetName, Question, Split

DATASET_NAMES: tuple[str, ...] = get_args(DatasetName)


def load_questions(
    datasets: list[str], split: str | None = None, *, root: Path | None = None
) -> list[Question]:
    """Questions from each named dataset, in file order, optionally filtered to one split."""
    if split is not None and split not in get_args(Split):
        raise ValueError(f"unknown split {split!r}; expected one of {get_args(Split)}")
    base = root or datasets_dir()
    out: list[Question] = []
    for name in datasets:
        if name not in DATASET_NAMES:
            raise ValueError(f"unknown dataset {name!r}; expected one of {DATASET_NAMES}")
        path = base / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `ledgerlens datasets build`")
        out.extend(q for q in read_jsonl(path) if split is None or q.split == split)
    return out
