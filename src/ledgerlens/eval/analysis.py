"""Error analysis: sort each failed question into a failure taxonomy.

Categories (schemas.FailureCategory):
- retrieval_miss: the evidence needed for the gold answer was never gathered.
- wrong_period: the answer is a value the evidence reports for a different period.
- table_misread: the gold value was in a gathered table but the answer used another cell.
- arithmetic: the inputs were gathered but the computed answer is wrong.
- hallucinated_citation: a citation does not support the answer it is attached to.
- bad_refusal: refused an answerable question, or answered an unanswerable one.
- other: none of the above explains it.

Classification is rule-based over what the run file recorded (answer, citations, gathered
chunk ids, scores) plus the chunk store and cached XBRL facts, so it costs no model calls
and is reproducible. Rules are ordered from most to least specific; the note says which
rule fired so a reader can audit it.
"""

from __future__ import annotations

import re
from collections import Counter

from ledgerlens.agent.graph import _matches, numbers_in
from ledgerlens.retrieval.retriever import get_store
from ledgerlens.schemas import Chunk, FailureCategory, QuestionResult, RunFile

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_COMPUTED = {"yoy_change", "multi_step"}


def is_failure(r: QuestionResult) -> bool:
    if r.error or r.prediction is None:
        return False
    s = r.scores
    for key in ("numeric_correct", "judge_correct"):
        if s.get(key) == 0.0:
            return True
    return s.get("refusal_correct") == 0.0


def _gathered_chunks(r: QuestionResult) -> list[Chunk]:
    if r.prediction is None:
        return []
    ids = [c for c in r.prediction.retrieved_chunk_ids if not c.startswith("xbrl:")]
    found = get_store().get_many(ids)
    return [found[c] for c in ids if c in found]


def _fact_values(r: QuestionResult) -> list[tuple[str, float]]:
    """(period, value) of XBRL facts the answer cited, parsed from fact ids and the quote."""
    out = []
    for cit in r.prediction.citations if r.prediction else []:
        if cit.chunk_id.startswith("xbrl:"):
            period = cit.chunk_id.split(":")[3]
            for v in numbers_in(cit.quote):
                out.append((period, v))
    return out


def classify(r: QuestionResult) -> tuple[FailureCategory, str]:
    pred = r.prediction
    assert pred is not None
    s = r.scores
    if not r.answerable:
        return "bad_refusal", "answered a question the filings cannot answer"
    if pred.refused or s.get("refusal_correct") == 0.0:
        return "bad_refusal", "refused although the filings answer it"

    chunks = _gathered_chunks(r)
    text = "\n".join(c.text for c in chunks)
    numbers = numbers_in(text) + [v for _, v in _fact_values(r)]
    gold, pred_value = r.gold_value, pred.answer_value
    if s.get("citation_precision") is not None and (s["citation_precision"] or 0) < 0.5:
        note = "most citations do not state the answer"
        if gold is not None and not _matches(gold, numbers):
            return "retrieval_miss", note + "; the gold value was never gathered"
        return "hallucinated_citation", note

    if gold is not None:
        gold_found = _matches(gold, numbers)
        if pred_value is not None and _matches(pred_value, numbers):
            for period, value in _fact_values(r):
                if _matches(pred_value, [value]) and not _matches(gold, [value]):
                    return "wrong_period", f"used the reported value for {period}"
            for chunk in chunks:
                if _matches(pred_value, numbers_in(chunk.text)) and gold_found:
                    years = set(_YEAR.findall(chunk.text))
                    if chunk.has_table:
                        if len(years) > 1:
                            return "wrong_period", "picked a cell from another period column"
                        return "table_misread", "picked another cell of a gathered table"
            if r.question_type in _COMPUTED:
                return "arithmetic", "computed from the wrong gathered inputs"
            return "wrong_period" if gold_found else "retrieval_miss", (
                "answer value is in the evidence but belongs to another line or period"
                if gold_found
                else "the gold value was never gathered"
            )
        if r.question_type in _COMPUTED:
            return "arithmetic", "inputs gathered but the computed result is wrong"
        if not gold_found:
            return "retrieval_miss", "the gold value was never gathered"
        return "other", "the gold value was gathered but the answer states another number"

    # Text answers (FinanceBench, judged).
    if (s.get("recall_at_10") or 0) == 0:
        return "retrieval_miss", "no labeled evidence page was gathered"
    if any(c.has_table for c in chunks):
        return "table_misread", "evidence page gathered (with tables) but the answer is wrong"
    return "other", "evidence page gathered but the answer is wrong"


def analyze(run: RunFile) -> Counter[str]:
    """Set failure_category and failure_note on every failed question, in place."""
    counts: Counter[str] = Counter()
    for r in run.results:
        if is_failure(r):
            category, note = classify(r)
            r.failure_category, r.failure_note = category, note
            counts[category] += 1
        else:
            r.failure_category, r.failure_note = None, None
    return counts
