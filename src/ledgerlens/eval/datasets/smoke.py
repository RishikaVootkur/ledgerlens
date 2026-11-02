"""Smoke set: 20 XBRL questions on the slice companies, for the thin slice and the CI gate.

Drawn only from `dev`-split XBRL questions about SLICE_TICKERS, so the held-out XBRL test
split never leaks into a set that gets tuned against. Mix: 12 single_fact, 4 yoy_change,
4 multi_step, picked round-robin across the three companies in SHA-256 order of
"smoke:{source qid}". Every smoke row is `dev` and keeps its source qid in
`metadata.source_qid`.
"""

from __future__ import annotations

from collections import defaultdict

from ledgerlens.eval.datasets.common import datasets_dir, read_jsonl, stable_hash, write_jsonl
from ledgerlens.logging import get_logger
from ledgerlens.schemas import Question
from ledgerlens.universe import SLICE_TICKERS

log = get_logger(__name__)

MIX = {"single_fact": 12, "yoy_change": 4, "multi_step": 4}


def _round_robin(pool: list[Question], n: int) -> list[Question]:
    by_ticker: dict[str, list[Question]] = defaultdict(list)
    for q in sorted(pool, key=lambda q: stable_hash(f"smoke:{q.qid}")):
        by_ticker[q.ticker or ""].append(q)
    order = [t for t in SLICE_TICKERS if by_ticker.get(t)]
    picked: list[Question] = []
    while len(picked) < n and any(by_ticker[t] for t in order):
        for t in order:
            if by_ticker[t] and len(picked) < n:
                picked.append(by_ticker[t].pop(0))
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} candidates for {n} smoke questions")
    return picked


def build(xbrl_questions: list[Question]) -> list[Question]:
    pool = [q for q in xbrl_questions if q.ticker in SLICE_TICKERS and q.split == "dev"]
    out: list[Question] = []
    for qtype, n in MIX.items():
        for src in _round_robin([q for q in pool if q.question_type == qtype], n):
            data = src.model_dump()
            data.update(
                qid=src.qid.replace("xbrl-", "smoke-", 1),
                dataset="smoke",
                split="dev",
                metadata={**src.metadata, "source_qid": src.qid},
            )
            out.append(Question.model_validate(data))
    return out


def generate() -> int:
    questions = build(read_jsonl(datasets_dir() / "xbrl.jsonl"))
    n = write_jsonl(datasets_dir() / "smoke.jsonl", questions)
    log.info("dataset_written", dataset="smoke", rows=n)
    return n
