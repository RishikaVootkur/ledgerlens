"""LLM-free retrieval evaluation: rank chunks for each question and score them against
labeled evidence. This powers the retrieval ablation (BM25 vs dense vs hybrid vs hybrid
plus rerank, and the chunking strategies) at zero model cost."""

from __future__ import annotations

from datetime import UTC, datetime

from ledgerlens.eval import metrics
from ledgerlens.eval.datasets.loader import load_questions
from ledgerlens.eval.runner import EvalConfig, select
from ledgerlens.eval.summary import summarize_by_dataset, summarize_scores
from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.retrieval.retriever import Retriever
from ledgerlens.schemas import Question, QuestionResult, RunFile, Split

K_VALUES = (1, 5, 10)


def retrieval_filters(question: Question, config: EvalConfig) -> SearchFilters:
    if config.scope == "document" and question.doc_name:
        return SearchFilters(doc_name=question.doc_name)
    if config.scope == "document" and question.ticker:
        return SearchFilters(ticker=question.ticker)
    return SearchFilters()


def score_ranking(ranked: list[set[str]], relevant: set[str]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for k in K_VALUES:
        scores[f"recall_at_{k}"] = metrics.recall_at_k(ranked, relevant, k)
    scores["mrr"] = metrics.mrr(ranked, relevant)
    scores["ndcg_at_10"] = metrics.ndcg_at_k(ranked, relevant, 10)
    return scores


def run_retrieval_eval(config: EvalConfig, split: Split, run_id: str | None = None) -> RunFile:
    questions = [
        q
        for q in select(load_questions(list(config.datasets), split), config)
        if q.answerable and metrics.evidence_keys(q)
    ]
    retriever = Retriever(config.strategy, config.method)
    results = []
    for q in questions:
        hits = retriever.search(q.question, max(K_VALUES), retrieval_filters(q, config))
        ranked = [metrics.chunk_keys(h.chunk) for h in hits]
        results.append(
            QuestionResult(
                qid=q.qid,
                dataset=q.dataset,
                question=q.question,
                question_type=q.question_type,
                answerable=q.answerable,
                ticker=q.ticker,
                fiscal_year=q.fiscal_year,
                gold=q.answer,
                prediction=None,
                scores=score_ranking(ranked, metrics.evidence_keys(q)),
            )
        )
    created = datetime.now(UTC)
    return RunFile(
        run_id=run_id or f"{created:%Y%m%d-%H%M%S}-{config.name}-{split}",
        created_at=created,
        config_name=config.name,
        config={**config.model_dump(mode="json"), "kind": "retrieval"},
        split=split,
        datasets=config.datasets,
        summary=summarize_scores(results),
        summary_by_dataset=summarize_by_dataset(results),
        results=results,
    )
