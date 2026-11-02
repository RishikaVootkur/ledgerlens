import pytest
from pydantic import ValidationError

from ledgerlens.schemas import (
    AgentAnswer,
    Chunk,
    Question,
    RunFile,
    Usage,
    make_chunk_id,
    make_filing_id,
)
from ledgerlens.universe import BY_TICKER, SLICE_TICKERS, UNIVERSE


def test_usage_adds():
    total = Usage(prompt_tokens=10, completion_tokens=2, llm_calls=1, cost_usd=0.5) + Usage(
        prompt_tokens=5, completion_tokens=1, llm_calls=1, cost_usd=0.25, latency_ms=3
    )
    assert (total.prompt_tokens, total.completion_tokens, total.llm_calls) == (15, 3, 2)
    assert total.cost_usd == 0.75 and total.latency_ms == 3


def test_universe_is_consistent():
    assert len(UNIVERSE) == 30
    assert len(BY_TICKER) == 30
    assert all(len(c.cik) == 10 and c.cik.isdigit() for c in UNIVERSE)
    assert set(SLICE_TICKERS) <= set(BY_TICKER)
    assert len({c.sector for c in UNIVERSE}) == 11
    assert len({c.cik for c in UNIVERSE}) == 30


def test_run_file_round_trips():
    q = Question(qid="smoke-1", dataset="smoke", split="dev", question="q?", question_type="t")
    run = RunFile(
        run_id="r1",
        created_at="2026-11-01T00:00:00+00:00",
        config_name="default",
        config={"k": 8},
        split="dev",
        datasets=["smoke"],
        summary={},
        results=[
            {
                "qid": q.qid,
                "dataset": q.dataset,
                "question": q.question,
                "gold": None,
                "prediction": AgentAnswer(answer="a"),
                "scores": {"numeric_correct": None},
            }
        ],
    )
    assert RunFile.model_validate_json(run.model_dump_json()) == run


def test_ids():
    assert make_filing_id("aapl", "10-K", 2025, "FY") == "AAPL-10K-2025FY"
    assert make_filing_id("KO", "10-Q", 2026, "Q2", "0000021344-26-000010") == "KO-10Q-2026Q2"
    assert make_filing_id("KO", "8-K", 2026, "Q2", "0000021344-26-000011").endswith("-000011")
    assert make_chunk_id("KO-10Q-2026Q2", "section", 7) == "KO-10Q-2026Q2:section:00007"


def test_question_rejects_contradictions():
    with pytest.raises(ValidationError):
        Question(
            qid="u",
            dataset="unanswerable",
            split="dev",
            question="q",
            answerable=False,
            answer="42",
            question_type="t",
        )
    with pytest.raises(ValidationError):
        Question(
            qid="x",
            dataset="xbrl",
            split="dev",
            question="q",
            answer="1",
            answer_value=1.0,
            question_type="t",
        )


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        Chunk.model_validate(
            {
                "chunk_id": "a",
                "filing_id": "f",
                "ticker": "T",
                "company": "C",
                "form": "10-K",
                "fiscal_year": 2025,
                "fiscal_period": "FY",
                "strategy": "fixed",
                "section": "s",
                "text": "t",
                "pg": 3,
            }
        )
