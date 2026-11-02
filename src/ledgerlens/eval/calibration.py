"""Judge calibration: agreement of the LLM judge with FinanceBench's human labels."""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel

from ledgerlens.eval.datasets.judge_calibration import CalibrationItem, load
from ledgerlens.eval.judge import JUDGE_PROMPT_VERSION, LABELS, Verdict, judge
from ledgerlens.eval.stats import agreement_accuracy, cohen_kappa, wilson_ci
from ledgerlens.schemas import MetricValue


class CalibrationResult(BaseModel):
    cid: str
    source_model: str
    human_label: str
    judge_label: str
    reason: str
    parse_error: bool


class CalibrationReport(BaseModel):
    created_at: datetime
    judge_model: str
    prompt_version: str
    split: str
    n: int
    accuracy: MetricValue
    kappa: MetricValue
    error_rate: MetricValue
    binary_accuracy: MetricValue  # CORRECT vs not, the way judge_correct is used in runs
    binary_kappa: MetricValue
    labels: list[str]
    confusion: list[list[int]]  # rows: human label, columns: judge label, order of `labels`
    per_label_recall: dict[str, float]
    parse_errors: int
    # Prompt v2 was tuned after reading answers to these questions; test items about other
    # questions form a clean subset that tuning never touched.
    clean_n: int | None = None
    clean_accuracy: MetricValue | None = None
    clean_kappa: MetricValue | None = None
    results: list[CalibrationResult]


def _kappa_ci(
    human: list[str], judged: list[str], n_resamples: int = 2000, seed: int = 0
) -> MetricValue:
    """Cohen's kappa with a percentile bootstrap interval over items."""
    rng = np.random.default_rng(seed)
    n = len(human)
    point = cohen_kappa(human, judged)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        stats.append(cohen_kappa([human[i] for i in idx], [judged[i] for i in idx]))
    low, high = np.percentile(stats, [2.5, 97.5])
    return MetricValue(value=point, ci_low=float(low), ci_high=float(high), n=n)


def _tuning_questions() -> set[str]:
    from ledgerlens.config import get_settings

    path = get_settings().eval_dir / "datasets" / "judge_calibration_tuning_questions.txt"
    if not path.exists():
        return set()
    return {x.strip() for x in path.read_text().splitlines() if x.strip() and not x.startswith("#")}


def calibrate(
    split: str = "test", model: str | None = None, concurrency: int = 2
) -> CalibrationReport:
    items: list[CalibrationItem] = load(split)
    kwargs = {"model": model} if model else {}

    def one(item: CalibrationItem) -> Verdict:
        return judge(item.question, item.gold_answer, item.model_answer, **kwargs)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        verdicts = list(pool.map(lambda i: contextvars.copy_context().run(one, i), items))
    human: list[str] = [i.human_label for i in items]
    judged: list[str] = [v.label for v in verdicts]
    agree = [1.0 if h == j else 0.0 for h, j in zip(human, judged, strict=True)]
    h_bin = [h == "CORRECT" for h in human]
    j_bin = [j == "CORRECT" for j in judged]
    labels: list[str] = list(LABELS)
    confusion = [
        [sum(1 for h, j in zip(human, judged, strict=True) if h == a and j == b) for b in labels]
        for a in labels
    ]
    from ledgerlens.llm import DEFAULT_JUDGE_MODEL

    tuned = _tuning_questions()
    clean = [k for k, i in enumerate(items) if i.financebench_id not in tuned]
    clean_h, clean_j = [human[k] for k in clean], [judged[k] for k in clean]

    return CalibrationReport(
        created_at=datetime.now(UTC),
        judge_model=model or DEFAULT_JUDGE_MODEL,
        prompt_version=JUDGE_PROMPT_VERSION,
        split=split,
        n=len(items),
        accuracy=wilson_ci(agree),
        kappa=_kappa_ci(human, judged),
        error_rate=wilson_ci([1.0 - a for a in agree]),
        binary_accuracy=wilson_ci(
            [1.0 if a == b else 0.0 for a, b in zip(h_bin, j_bin, strict=True)]
        ),
        binary_kappa=_kappa_ci([str(x) for x in h_bin], [str(x) for x in j_bin]),
        labels=labels,
        confusion=confusion,
        per_label_recall={
            lab: (confusion[k][k] / max(1, sum(confusion[k]))) for k, lab in enumerate(labels)
        },
        parse_errors=sum(v.parse_error for v in verdicts),
        clean_n=len(clean) if tuned else None,
        clean_accuracy=wilson_ci(
            [1.0 if h == j else 0.0 for h, j in zip(clean_h, clean_j, strict=True)]
        )
        if tuned and clean
        else None,
        clean_kappa=_kappa_ci(clean_h, clean_j) if tuned and len(set(clean_h)) > 1 else None,
        results=[
            CalibrationResult(
                cid=i.cid,
                source_model=i.source_model,
                human_label=i.human_label,
                judge_label=v.label,
                reason=v.reason,
                parse_error=v.parse_error,
            )
            for i, v in zip(items, verdicts, strict=True)
        ],
    )


__all__ = ["CalibrationReport", "agreement_accuracy", "calibrate"]
