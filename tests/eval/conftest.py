"""Builders for run files used across the eval tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from ledgerlens.schemas import AgentAnswer, QuestionResult, RunFile, Usage

ResultBuilder = Callable[..., QuestionResult]
RunBuilder = Callable[..., RunFile]


def _result(
    qid: str,
    scores: dict[str, float | None],
    *,
    dataset: str = "xbrl",
    latency_ms: float = 100.0,
    wall_ms: float | None = None,
    tokens: tuple[int, int] = (100, 10),
    cost: float = 0.001,
    error: str | None = None,
    answered: bool = True,
) -> QuestionResult:
    prediction = (
        AgentAnswer(
            answer="42",
            usage=Usage(
                prompt_tokens=tokens[0],
                completion_tokens=tokens[1],
                llm_calls=1,
                cost_usd=cost,
                latency_ms=latency_ms,
            ),
        )
        if answered
        else None
    )
    return QuestionResult(
        qid=qid,
        dataset=dataset,  # type: ignore[arg-type]
        question=f"question {qid}?",
        gold="42",
        prediction=prediction,
        wall_ms=wall_ms,
        scores=scores,
        error=error,
    )


def _run(run_id: str, results: list[QuestionResult], **kwargs: Any) -> RunFile:
    return RunFile(
        run_id=run_id,
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
        config_name=kwargs.pop("config_name", f"cfg-{run_id}"),
        config={},
        split="dev",
        datasets=["xbrl"],
        summary={},
        results=results,
        **kwargs,
    )


@pytest.fixture
def make_result() -> ResultBuilder:
    return _result


@pytest.fixture
def make_run() -> RunBuilder:
    return _run
