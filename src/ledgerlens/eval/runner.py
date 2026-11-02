"""Run the agent over an eval split, score every answer and write a run file.

A run is fully described by its YAML config, so `eval compare` can say exactly what
changed between two runs. With `--record-cache`, every LLM and tool response the run
touched is copied into `eval/cache`, which is what the CI gate replays.
"""

from __future__ import annotations

import contextvars
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from ledgerlens.agent.graph import AgentConfig, quote_supported, run_agent
from ledgerlens.agent.tools import EvidenceItem, EvidenceRegistry, RetrievalConfig
from ledgerlens.config import get_settings
from ledgerlens.eval import metrics
from ledgerlens.eval.datasets.loader import load_questions
from ledgerlens.eval.judge import judge as run_judge
from ledgerlens.eval.summary import summarize_by_dataset, summarize_scores
from ledgerlens.llm import DEFAULT_ANSWER_MODEL, DEFAULT_JUDGE_MODEL
from ledgerlens.llm.cache import touched_entries
from ledgerlens.logging import get_logger
from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.retrieval.retriever import Method
from ledgerlens.schemas import (
    AgentAnswer,
    ChunkStrategy,
    DatasetName,
    Question,
    QuestionResult,
    RunFile,
    Split,
    Usage,
)

log = get_logger(__name__)


class EvalConfig(BaseModel):
    """One experiment. Every field that can change an answer lives here."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    model: str = DEFAULT_ANSWER_MODEL
    strategy: ChunkStrategy = "section"
    method: Method = "hybrid"
    k: int = 6
    index_version: str = "auto"
    max_rounds: int = 3
    xbrl_tool: bool = True
    verify_numbers: bool = True
    self_check: bool = True
    datasets: list[DatasetName] = ["smoke"]
    # document: FinanceBench questions are searched within their own document, the
    # FinanceBench "single document" setting. open: no filter, the agent must find it.
    scope: Literal["open", "document"] = "document"
    limit: int | None = None
    qids: list[str] | None = None
    concurrency: int = 2
    # LLM judge for FinanceBench answers (judge_correct). Different family from the
    # answer models; calibrated in eval/reports/judge_calibration_*.json.
    judge: bool = True
    judge_model: str = DEFAULT_JUDGE_MODEL

    def agent_config(self) -> AgentConfig:
        return AgentConfig(
            model=self.model,
            retrieval=RetrievalConfig(
                strategy=self.strategy,
                method=self.method,
                k=self.k,
                index_version=self.index_version,
            ),
            max_rounds=self.max_rounds,
            xbrl_tool=self.xbrl_tool,
            verify_numbers=self.verify_numbers,
            self_check=self.self_check,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> EvalConfig:
        return cls.model_validate(yaml.safe_load(path.read_text()))


# ------------------------------------------------------------------ data


def select(questions: list[Question], config: EvalConfig) -> list[Question]:
    if config.qids is not None:
        wanted = set(config.qids)
        questions = [q for q in questions if q.qid in wanted]
    if config.limit is not None:
        questions = questions[: config.limit]
    return questions


# ------------------------------------------------------------------ scoring


def item_keys(item: EvidenceItem) -> set[str]:
    """Relevance keys of an evidence item, matching `metrics.chunk_keys` for chunks."""
    docs = [d for d in (item.filing_id, item.doc_name) if d]
    keys = {item.source_id, *docs}
    if item.page is not None:
        last = max(item.page, item.page_end or item.page)
        keys |= {metrics.page_key(d, p) for d in docs for p in range(item.page, last + 1)}
    return keys


def _contains_value(text: str, value: float) -> bool:
    from ledgerlens.agent.graph import _grounded, numbers_in

    return _grounded(value, numbers_in(text)) if abs(value) > 31 else False


def score(
    question: Question, answer: AgentAnswer, registry: EvidenceRegistry
) -> dict[str, float | None]:
    """Per-question scores. Keys match eval/gate.yaml.

    citation_precision here is deterministic: a citation supports a numeric answer
    when its source states the headline number (at any reporting scale), and a text
    answer when its quote appears verbatim in the source. The LLM judge adds a
    semantic support check for text answers.
    """
    scores: dict[str, float | None] = {
        "numeric_correct": metrics.numeric_correct(answer, question),
        "refusal_correct": 1.0 if metrics.refusal_correct(answer, question.answerable) else 0.0,
    }
    by_source = {item.source_id: item for item in registry.items.values()}
    supported: list[bool] = []
    cited: set[str] = set()
    for cit in answer.citations:
        item = by_source.get(cit.chunk_id)
        if item is None:
            supported.append(False)
            continue
        cited |= item_keys(item)
        if answer.answer_value is not None:
            supported.append(_contains_value(item.text, answer.answer_value))
        else:
            supported.append(bool(cit.quote) and quote_supported(cit.quote, item.text))
    relevant = metrics.evidence_keys(question)
    answered = question.answerable and not answer.refused
    scores["citation_precision"] = metrics.citation_precision(supported) if answered else None
    scores["citation_recall"] = metrics.citation_recall(cited, relevant) if answered else None
    retrieved = [item_keys(i) for i in registry.items.values() if i.kind == "chunk"]
    if question.answerable and relevant:
        scores["recall_at_5"] = metrics.recall_at_k(retrieved, relevant, 5)
        scores["recall_at_10"] = metrics.recall_at_k(retrieved, relevant, 10)
        scores["mrr"] = metrics.mrr(retrieved, relevant)
        scores["ndcg_at_10"] = metrics.ndcg_at_k(retrieved, relevant, 10)
    return scores


# ------------------------------------------------------------------ running


def _filters(question: Question, config: EvalConfig) -> SearchFilters:
    if config.scope == "document" and question.dataset == "financebench" and question.doc_name:
        return SearchFilters(doc_name=question.doc_name)
    return SearchFilters()


def run_one(question: Question, config: EvalConfig) -> QuestionResult:
    registry = EvidenceRegistry()
    started = time.perf_counter()
    answer: AgentAnswer | None = None
    error: str | None = None
    scores: dict[str, float | None] = {}
    try:
        answer = run_agent(
            question.question,
            config.agent_config(),
            _filters(question, config),
            registry=registry,
        )
        scores = score(question, answer, registry)
        if config.judge and question.dataset == "financebench" and question.answer:
            verdict = run_judge(
                question.question, question.answer, answer.answer, model=config.judge_model
            )
            scores["judge_correct"] = 1.0 if verdict.label == "CORRECT" else 0.0
    except Exception as exc:  # recorded per question; the run continues
        error = f"{type(exc).__name__}: {str(exc)[:500]}"
        log.warning("eval.question_failed", qid=question.qid, error=error)
    return QuestionResult(
        qid=question.qid,
        dataset=question.dataset,
        question=question.question,
        question_type=question.question_type,
        answerable=question.answerable,
        ticker=question.ticker,
        fiscal_year=question.fiscal_year,
        gold=question.answer,
        gold_value=question.answer_value,
        gold_unit=question.answer_unit,
        prediction=answer,
        wall_ms=round((time.perf_counter() - started) * 1000, 1),
        scores=scores,
        error=error,
    )


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=get_settings().eval_dir.parent,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def run_eval(
    config: EvalConfig,
    split: Split,
    *,
    run_id: str | None = None,
    progress: Callable[[int, int, QuestionResult], None] | None = None,
) -> RunFile:
    questions = select(load_questions(list(config.datasets), split), config)
    results: list[QuestionResult | None] = [None] * len(questions)
    done = 0

    def task(i: int) -> None:
        nonlocal done
        results[i] = run_one(questions[i], config)
        done += 1
        if progress:
            progress(done, len(questions), results[i])  # type: ignore[arg-type]

    # Each worker gets a copy of the caller's context so usage tracking works per call.
    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
        futures = [
            pool.submit(contextvars.copy_context().run, task, i) for i in range(len(questions))
        ]
        for f in futures:
            f.result()
    finished = [r for r in results if r is not None]
    created = datetime.now(UTC)
    total = sum((r.prediction.usage for r in finished if r.prediction), start=Usage())
    return RunFile(
        run_id=run_id or f"{created:%Y%m%d-%H%M%S}-{config.name}-{split}",
        created_at=created,
        git_sha=_git_sha(),
        config_name=config.name,
        config=config.model_dump(mode="json"),
        split=split,
        datasets=config.datasets,
        summary=summarize_scores(finished),
        summary_by_dataset=summarize_by_dataset(finished),
        usage_total=total,
        results=finished,
    )


def write_run(run: RunFile, out: Path | None = None) -> Path:
    path = out or get_settings().eval_dir / "runs" / f"{run.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=1))
    return path


def record_cache() -> int:
    """Copy every cache entry touched in this process into the committed eval/cache."""
    s = get_settings()
    dest_root = s.eval_dir / "cache"
    copied = 0
    for namespace, key in sorted(touched_entries()):
        if namespace == "embed":
            continue  # query vectors are not needed for replay (tool results are cached)
        src = s.cache_dir / namespace / key[:2] / f"{key}.json"
        dest = dest_root / namespace / key[:2] / f"{key}.json"
        if src.exists() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(src.read_text())
            copied += 1
    return copied
