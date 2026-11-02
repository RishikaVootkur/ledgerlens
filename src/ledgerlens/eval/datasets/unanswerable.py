"""Unanswerable set: questions the corpus cannot answer, to measure refusal and hallucination.

Questions are handwritten in `seeds/unanswerable.yaml` (one entry per question, with a
stable id, a reason category and a `why` note). Each becomes a `Question` with
`answerable=False`, no gold answer, and `question_type` set to the reason category.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ledgerlens.eval.datasets.common import assign_splits, datasets_dir, write_jsonl
from ledgerlens.logging import get_logger
from ledgerlens.schemas import Question
from ledgerlens.universe import BY_TICKER

log = get_logger(__name__)

SEEDS = Path(__file__).parent / "seeds" / "unanswerable.yaml"

CATEGORIES = (
    "future_period",
    "not_in_corpus_company",
    "not_disclosed",
    "wrong_form_or_period",
    "false_premise",
    "nonfinancial_trivia",
)
_FIELDS = {"id", "category", "question", "why", "ticker", "company", "fiscal_year", "fiscal_period"}


def load_seeds(path: Path = SEEDS) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of seed entries")
    return data


def to_question(seed: dict[str, Any]) -> Question:
    unknown = set(seed) - _FIELDS
    if unknown:
        raise ValueError(f"{seed.get('id')}: unknown seed fields {sorted(unknown)}")
    if seed["category"] not in CATEGORIES:
        raise ValueError(f"{seed['id']}: unknown category {seed['category']!r}")
    ticker = seed.get("ticker")
    company = seed.get("company") or (BY_TICKER[ticker].name if ticker in BY_TICKER else None)
    in_universe = ticker in BY_TICKER
    if seed["category"] == "not_in_corpus_company" and in_universe:
        raise ValueError(f"{seed['id']}: {ticker} is in the universe")
    return Question(
        qid=seed["id"],
        dataset="unanswerable",
        split="test",  # replaced by assign_splits
        question=seed["question"],
        answerable=False,
        ticker=ticker,
        company=company,
        fiscal_year=seed.get("fiscal_year"),
        fiscal_period=seed.get("fiscal_period"),
        question_type=seed["category"],
        metadata={"why": seed["why"], "in_universe": in_universe},
    )


def build(seeds: list[dict[str, Any]] | None = None) -> list[Question]:
    questions = [to_question(s) for s in (seeds if seeds is not None else load_seeds())]
    qids = [q.qid for q in questions]
    if len(qids) != len(set(qids)):
        raise ValueError("duplicate unanswerable ids")
    return assign_splits(questions)


def generate() -> int:
    n = write_jsonl(datasets_dir() / "unanswerable.jsonl", build())
    log.info("dataset_written", dataset="unanswerable", rows=n)
    return n
