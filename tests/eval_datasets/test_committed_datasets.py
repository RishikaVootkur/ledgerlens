"""Checks on the JSONL files committed under eval/datasets/ (no network)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from ledgerlens.eval.datasets.common import assign_splits
from ledgerlens.eval.datasets.smoke import underlying_key
from ledgerlens.eval.datasets.unanswerable import CATEGORIES, build, load_seeds, to_question
from ledgerlens.schemas import Question
from ledgerlens.universe import BY_TICKER, SLICE_TICKERS

DATA = Path(__file__).resolve().parents[2] / "eval" / "datasets"
NAMES = ("financebench", "xbrl", "unanswerable", "smoke")


def _load(name: str) -> list[Question]:
    lines = (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [Question.model_validate(json.loads(line)) for line in lines if line.strip()]


@pytest.fixture(scope="module")
def datasets() -> dict[str, list[Question]]:
    return {name: _load(name) for name in NAMES}


def test_every_row_validates_and_qids_are_unique(datasets):
    all_qids = [q.qid for qs in datasets.values() for q in qs]
    assert len(all_qids) == len(set(all_qids))
    for name, qs in datasets.items():
        assert {q.dataset for q in qs} == {name}


def test_sizes(datasets):
    assert len(datasets["financebench"]) == 150
    assert len(datasets["xbrl"]) >= 200
    assert len(datasets["unanswerable"]) >= 50
    assert len(datasets["smoke"]) == 20


@pytest.mark.parametrize("name", ["financebench", "xbrl", "unanswerable"])
def test_committed_splits_follow_the_rule(datasets, name):
    qs = datasets[name]
    expected = {q.qid: q.split for q in assign_splits(qs)}
    assert {q.qid: q.split for q in qs} == expected
    dev_share = sum(q.split == "dev" for q in qs) / len(qs)
    assert 0.37 <= dev_share <= 0.43


def test_xbrl_rows_cite_a_filing_and_concept(datasets):
    for q in datasets["xbrl"]:
        assert q.answer_value is not None and q.answer_unit is not None
        assert q.ticker in BY_TICKER
        assert q.evidence and all(e.filing_id and e.xbrl_concept for e in q.evidence)
        assert all(e.filing_id.startswith(f"{q.ticker}-") for e in q.evidence)
        expected_tol = 0.01 if q.question_type == "single_fact" else 0.02
        assert q.rel_tolerance == expected_tol
    types = Counter(q.question_type for q in datasets["xbrl"])
    assert set(types) == {"single_fact", "yoy_change", "multi_step"}


def test_smoke_is_dev_slice_drawn_from_xbrl_dev(datasets):
    smoke = datasets["smoke"]
    xbrl = {q.qid: q for q in datasets["xbrl"]}
    assert Counter(q.question_type for q in smoke) == {
        "single_fact": 12,
        "yoy_change": 4,
        "multi_step": 4,
    }
    for q in smoke:
        assert q.split == "dev" and q.ticker in SLICE_TICKERS
        src = xbrl[str(q.metadata["source_qid"])]
        assert src.split == "dev"
        assert (src.question, src.answer_value) == (q.question, q.answer_value)
    assert {q.ticker for q in smoke} == set(SLICE_TICKERS)


def test_unanswerable_rows(datasets):
    qs = datasets["unanswerable"]
    assert all(not q.answerable and q.answer is None and q.answer_value is None for q in qs)
    counts = Counter(q.question_type for q in qs)
    assert set(counts) == set(CATEGORIES)
    assert min(counts.values()) >= 5


def test_financebench_evidence_pages_are_one_based(datasets):
    pages = [e.page for q in datasets["financebench"] for e in q.evidence]
    assert all(p is not None and p >= 1 for p in pages)


def test_unanswerable_committed_file_matches_seeds(datasets):
    assert [q.model_dump() for q in build()] == [q.model_dump() for q in datasets["unanswerable"]]


def test_seed_validation():
    seed = dict(load_seeds()[0])
    with pytest.raises(ValueError, match="unknown category"):
        to_question({**seed, "category": "nonsense"})
    with pytest.raises(ValueError, match="in the universe"):
        to_question({**seed, "category": "not_in_corpus_company", "ticker": "AAPL"})
    with pytest.raises(ValueError, match="unknown seed fields"):
        to_question({**seed, "answer": "42"})


def test_smoke_questions_read_distinct_facts(datasets):
    smoke = datasets["smoke"]
    assert len({underlying_key(q) for q in smoke}) == len(smoke)
    yoy = [q for q in smoke if q.question_type == "yoy_change"]
    assert len({underlying_key(q) for q in yoy}) == 4


def test_abs_and_pct_variants_share_an_underlying_key(datasets):
    xbrl = {q.qid: q for q in datasets["xbrl"]}
    stem = "xbrl-aapl-10k2025fy-operating_income-yoy"
    assert underlying_key(xbrl[f"{stem}-abs"]) == underlying_key(xbrl[f"{stem}-pct"])
