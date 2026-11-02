from ledgerlens.schemas import AgentAnswer, Question, RunFile, Usage
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
    assert len({c.sector for c in UNIVERSE}) >= 10


def test_run_file_round_trips():
    q = Question(qid="smoke-1", dataset="smoke", split="dev", question="q?", question_type="t")
    run = RunFile(
        run_id="r1",
        created_at="2026-11-01T00:00:00Z",
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
