from collections import Counter

import pytest

from ledgerlens.eval.datasets.common import assign_splits, write_jsonl
from ledgerlens.eval.datasets.loader import load_questions
from ledgerlens.schemas import Question


def _questions(n: int, qtype: str) -> list[Question]:
    return [
        Question(
            qid=f"t-{qtype}-{i:03d}",
            dataset="xbrl",
            split="test",
            question="q",
            question_type=qtype,
        )
        for i in range(n)
    ]


def test_splits_are_deterministic_and_order_independent():
    qs = _questions(50, "a") + _questions(30, "b")
    first = {q.qid: q.split for q in assign_splits(qs)}
    again = {q.qid: q.split for q in assign_splits(list(reversed(qs)))}
    assert first == again


def test_splits_are_40_60_within_each_type():
    qs = _questions(50, "a") + _questions(30, "b") + _questions(7, "c")
    counts = Counter((q.question_type, q.split) for q in assign_splits(qs))
    assert counts[("a", "dev")] == 20 and counts[("a", "test")] == 30
    assert counts[("b", "dev")] == 12 and counts[("b", "test")] == 18
    assert counts[("c", "dev")] == 3 and counts[("c", "test")] == 4


def test_assign_splits_keeps_input_order():
    qs = _questions(10, "a")
    assert [q.qid for q in assign_splits(qs)] == [q.qid for q in qs]


def test_loader_filters_by_split(tmp_path):
    write_jsonl(tmp_path / "xbrl.jsonl", assign_splits(_questions(10, "a")))
    assert len(load_questions(["xbrl"], None, root=tmp_path)) == 10
    assert len(load_questions(["xbrl"], "dev", root=tmp_path)) == 4
    assert len(load_questions(["xbrl"], "test", root=tmp_path)) == 6


def test_loader_rejects_unknown_names(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        load_questions(["nope"], None, root=tmp_path)
    with pytest.raises(ValueError, match="unknown split"):
        load_questions(["xbrl"], "train", root=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_questions(["smoke"], None, root=tmp_path)
