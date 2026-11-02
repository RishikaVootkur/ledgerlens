"""Shared data contracts.

Ingestion, retrieval, the agent, the eval lab and the web UI all exchange these
shapes. Change them only in a dedicated PR, because several modules depend on them.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FormType = Literal["10-K", "10-Q", "8-K", "EARNINGS", "OTHER"]
FiscalPeriod = Literal["FY", "Q1", "Q2", "Q3", "Q4"]
ChunkStrategy = Literal["fixed", "section"]
BlockKind = Literal["heading", "text", "table"]
FailureCategory = Literal[
    "retrieval_miss",
    "table_misread",
    "arithmetic",
    "wrong_period",
    "hallucinated_citation",
    "bad_refusal",
    "other",
]

SCHEMA_VERSION = 1


class Contract(BaseModel):
    """Base for every contract: misspelled or unknown fields are errors, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


def make_filing_id(
    ticker: str, form: str, fiscal_year: int, fiscal_period: str, accession: str | None = None
) -> str:
    """Build the stable filing id, e.g. ``AAPL-10K-2025FY``.

    The form loses its dash (10-K -> 10K). 8-K, EARNINGS and OTHER documents can repeat
    within a period, so their ids end with the accession number when one is known.
    """
    base = f"{ticker.upper()}-{form.replace('-', '').upper()}-{fiscal_year}{fiscal_period}"
    if form not in ("10-K", "10-Q") and accession:
        return f"{base}-{accession}"
    return base


def make_chunk_id(filing_id: str, strategy: str, seq: int) -> str:
    return f"{filing_id}:{strategy}:{seq:05d}"


# ---------------------------------------------------------------- ingestion


class FilingRef(Contract):
    """One filing document. `filing_id` is stable across reruns."""

    filing_id: str = Field(description="Built by make_filing_id, e.g. AAPL-10K-2025FY")
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


class Block(Contract):
    """A parsed unit of a filing in reading order. Tables are GitHub Markdown."""

    kind: BlockKind
    text: str
    section: str = Field(description="Nearest heading path, e.g. 'Item 7. MD&A > Liquidity'")
    page: int | None = Field(default=None, description="1-based page; PDFs exact, HTML estimated")
    table_id: str | None = None


class ParsedFiling(Contract):
    filing: FilingRef
    blocks: list[Block]


class Chunk(Contract):
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


class Evidence(Contract):
    """Where the gold answer lives. Used for retrieval metrics and citation recall.

    FinanceBench evidence has doc_name + page. XBRL evidence has filing_id + concept, and
    optionally the section or chunk ids once the value has been located in the text.
    HTML page numbers are estimates, so page-level matching is exact only for PDFs.
    """

    doc_name: str | None = None
    filing_id: str | None = None
    page: int | None = Field(default=None, description="1-based page in the source document")
    text: str | None = Field(default=None, description="Gold evidence passage, if labeled")
    xbrl_concept: str | None = Field(default=None, description="e.g. us-gaap:Revenues")
    section: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)


class Question(Contract):
    qid: str = Field(description="Stable id, e.g. fb-00123, xbrl-0042, unans-007")
    dataset: DatasetName
    split: Split
    question: str
    answerable: bool = True
    answer: str | None = Field(
        default=None, description="Gold answer as text; None if unanswerable"
    )
    answer_value: float | None = Field(
        default=None,
        description="Numeric gold in base units: USD not millions; percent as 12.5 for 12.5%",
    )
    answer_unit: AnswerUnit | None = None
    rel_tolerance: float = Field(default=0.01, description="Relative tolerance for numeric match")
    abs_tolerance: float = Field(default=0.0, description="Absolute tolerance, for golds near 0")
    ticker: str | None = None
    company: str | None = None
    fiscal_year: int | None = None
    fiscal_period: FiscalPeriod | None = None
    form: FormType | None = None
    doc_name: str | None = None
    question_type: str = Field(description="e.g. single_fact, yoy_change, multi_step, domain")
    evidence: list[Evidence] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> Question:
        if not self.answerable and (self.answer is not None or self.answer_value is not None):
            raise ValueError(f"{self.qid}: unanswerable questions carry no gold answer")
        if self.answer_value is not None and self.answer_unit is None:
            raise ValueError(f"{self.qid}: answer_value needs answer_unit")
        return self


# ---------------------------------------------------------------- agent output


class Citation(Contract):
    chunk_id: str
    filing_id: str
    section: str
    page: int | None = None
    page_end: int | None = None
    quote: str = Field(default="", description="Short supporting span from the chunk")


class Usage(Contract):
    """Token, cost and latency totals. `latency_ms` sums call latencies (not wall time)."""

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


class AgentAnswer(Contract):
    answer: str
    refused: bool = False
    answer_value: float | None = Field(default=None, description="Headline number in base units")
    answer_unit: AnswerUnit | None = None
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list, description="Agent step names in order")
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------- run files


class MetricValue(Contract):
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int


class QuestionResult(Contract):
    qid: str
    dataset: DatasetName
    question: str
    question_type: str | None = None
    answerable: bool = True
    ticker: str | None = None
    fiscal_year: int | None = None
    gold: str | None
    gold_value: float | None = None
    gold_unit: AnswerUnit | None = None
    prediction: AgentAnswer | None
    wall_ms: float | None = Field(default=None, description="End-to-end time for this question")
    scores: dict[str, float | None] = Field(
        description="Per-question metric values; None when a metric does not apply"
    )
    error: str | None = Field(default=None, description="Exception text if the run crashed")
    failure_category: FailureCategory | None = None
    failure_note: str | None = None


class RunFile(Contract):
    """One eval run, written to eval/runs/{run_id}.json and read by `eval compare` and the UI."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    created_at: datetime
    git_sha: str | None = None
    config_name: str
    config: dict[str, object]
    split: Split
    datasets: list[DatasetName]
    summary: dict[str, MetricValue]
    summary_by_dataset: dict[str, dict[str, MetricValue]] = Field(default_factory=dict)
    usage_total: Usage = Field(default_factory=Usage)
    results: list[QuestionResult]
