"""Shared data contracts.

Ingestion, retrieval, the agent, the eval lab and the web UI all exchange these
shapes. Change them only in a dedicated PR, because several modules depend on them.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

FormType = Literal["10-K", "10-Q", "8-K", "EARNINGS", "OTHER"]
FiscalPeriod = Literal["FY", "Q1", "Q2", "Q3", "Q4"]
ChunkStrategy = Literal["fixed", "section"]
BlockKind = Literal["heading", "text", "table"]


# ---------------------------------------------------------------- ingestion


class FilingRef(BaseModel):
    """One filing document. `filing_id` is stable across reruns."""

    filing_id: str = Field(
        description="'{ticker}-{form}-{fiscal_year}{fiscal_period}', e.g. AAPL-10K-2025FY"
    )
    cik: str = Field(description="10-digit zero-padded CIK")
    ticker: str
    company: str
    form: FormType
    fiscal_year: int
    fiscal_period: FiscalPeriod
    period_end: date | None = None
    filed: date | None = None
    accession: str | None = Field(default=None, description="EDGAR accession, dashes kept")
    source_url: str
    source_format: Literal["html", "pdf"]
    doc_name: str | None = Field(
        default=None, description="FinanceBench doc_name, e.g. 3M_2018_10K"
    )


class Block(BaseModel):
    """A parsed unit of a filing in reading order. Tables are GitHub Markdown."""

    kind: BlockKind
    text: str
    section: str = Field(description="Nearest heading path, e.g. 'Item 7. MD&A > Liquidity'")
    page: int | None = Field(default=None, description="1-based page; PDFs exact, HTML estimated")
    table_id: str | None = None


class ParsedFiling(BaseModel):
    filing: FilingRef
    blocks: list[Block]


class Chunk(BaseModel):
    """A retrievable unit. Metadata is denormalized so filters need no joins."""

    chunk_id: str = Field(description="'{filing_id}:{strategy}:{seq:05d}'")
    filing_id: str
    ticker: str
    company: str
    form: FormType
    fiscal_year: int
    fiscal_period: FiscalPeriod
    doc_name: str | None = None
    strategy: ChunkStrategy
    section: str
    page: int | None = None
    page_end: int | None = None
    text: str
    has_table: bool = False
    table_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- eval data

DatasetName = Literal["financebench", "xbrl", "unanswerable", "smoke"]
Split = Literal["dev", "test"]
AnswerUnit = Literal["USD", "shares", "percent", "ratio", "per_share", "count", "text"]


class Evidence(BaseModel):
    """Where the gold answer lives. Used for retrieval metrics and citation recall."""

    doc_name: str | None = None
    filing_id: str | None = None
    page: int | None = Field(default=None, description="1-based page in the source document")
    text: str | None = Field(default=None, description="Gold evidence passage, if labeled")
    xbrl_concept: str | None = Field(default=None, description="e.g. us-gaap:Revenues")


class Question(BaseModel):
    qid: str = Field(description="Stable id, e.g. fb-00123, xbrl-0042, unans-007")
    dataset: DatasetName
    split: Split
    question: str
    answerable: bool = True
    answer: str | None = Field(
        default=None, description="Gold answer as text; None if unanswerable"
    )
    answer_value: float | None = Field(default=None, description="Numeric gold in base units")
    answer_unit: AnswerUnit | None = None
    rel_tolerance: float = Field(default=0.01, description="Relative tolerance for numeric match")
    ticker: str | None = None
    company: str | None = None
    fiscal_year: int | None = None
    fiscal_period: FiscalPeriod | None = None
    form: FormType | None = None
    doc_name: str | None = None
    question_type: str = Field(description="e.g. single_fact, yoy_change, multi_step, domain")
    evidence: list[Evidence] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


# ---------------------------------------------------------------- agent output


class Citation(BaseModel):
    chunk_id: str
    filing_id: str
    section: str
    page: int | None = None
    quote: str = Field(default="", description="Short supporting span from the chunk")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = Field(default=0.0, description="List-price equivalent, even when free")
    latency_ms: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_ms=self.latency_ms + other.latency_ms,
        )


class AgentAnswer(BaseModel):
    answer: str
    refused: bool = False
    answer_value: float | None = Field(default=None, description="Parsed headline number, if any")
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list, description="Agent step names in order")
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------- run files


class MetricValue(BaseModel):
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int


class QuestionResult(BaseModel):
    qid: str
    dataset: DatasetName
    question: str
    gold: str | None
    prediction: AgentAnswer | None
    scores: dict[str, float | None] = Field(
        description="Per-question metric values; None when a metric does not apply"
    )
    error: str | None = Field(default=None, description="Exception text if the run crashed")
    failure_category: str | None = Field(
        default=None,
        description="retrieval_miss | table_misread | arithmetic | wrong_period | "
        "hallucinated_citation | bad_refusal | other",
    )


class RunFile(BaseModel):
    """One eval run, written to eval/runs/{run_id}.json and read by `eval compare` and the UI."""

    run_id: str
    created_at: str
    git_sha: str | None = None
    config_name: str
    config: dict[str, object]
    split: Split
    datasets: list[DatasetName]
    summary: dict[str, MetricValue]
    results: list[QuestionResult]
