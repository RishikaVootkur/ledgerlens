// TypeScript mirrors of src/ledgerlens/schemas.py plus the HTTP contract
// between the FastAPI backend and this UI. Keep field names identical to the
// Pydantic models; optional Python fields (`X | None = None`) are `X | null`
// and may also be omitted.

// ---------------------------------------------------------------- literals

export type FormType = "10-K" | "10-Q" | "8-K" | "EARNINGS" | "OTHER";
export type FiscalPeriod = "FY" | "Q1" | "Q2" | "Q3" | "Q4";
export type ChunkStrategy = "fixed" | "section";
export type DatasetName = "financebench" | "xbrl" | "unanswerable" | "smoke";
export type Split = "dev" | "test";
export type AnswerUnit = "USD" | "shares" | "percent" | "ratio" | "per_share" | "count" | "text";

/** RunFile.schema_version this UI understands. */
export const SCHEMA_VERSION = 1;

export const FAILURE_CATEGORIES = [
  "retrieval_miss",
  "table_misread",
  "arithmetic",
  "wrong_period",
  "hallucinated_citation",
  "bad_refusal",
  "other",
] as const;
export type FailureCategory = (typeof FAILURE_CATEGORIES)[number];

// ---------------------------------------------------------------- schemas.py

export interface Chunk {
  chunk_id: string;
  filing_id: string;
  ticker: string;
  company: string;
  form: FormType;
  fiscal_year: number;
  fiscal_period: FiscalPeriod;
  doc_name?: string | null;
  strategy: ChunkStrategy;
  section: string;
  page?: number | null;
  page_end?: number | null;
  /** May contain GitHub Markdown tables. */
  text: string;
  has_table: boolean;
  table_ids: string[];
}

export interface Citation {
  chunk_id: string;
  filing_id: string;
  section: string;
  page?: number | null;
  page_end?: number | null;
  /** Short supporting span from the chunk. */
  quote: string;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  llm_calls: number;
  /** List-price equivalent, even when the provider was free. */
  cost_usd: number;
  latency_ms: number;
}

export interface AgentAnswer {
  answer: string;
  refused: boolean;
  /** Headline number in base units. */
  answer_value?: number | null;
  answer_unit?: AnswerUnit | null;
  citations: Citation[];
  retrieved_chunk_ids: string[];
  steps: string[];
  usage: Usage;
}

export interface MetricValue {
  value: number;
  ci_low?: number | null;
  ci_high?: number | null;
  n: number;
}

export interface QuestionResult {
  qid: string;
  dataset: DatasetName;
  question: string;
  question_type?: string | null;
  answerable: boolean;
  ticker?: string | null;
  fiscal_year?: number | null;
  gold: string | null;
  gold_value?: number | null;
  gold_unit?: AnswerUnit | null;
  prediction: AgentAnswer | null;
  /** End-to-end time for this question. */
  wall_ms?: number | null;
  /** Per-question metric values; null when a metric does not apply. */
  scores: Record<string, number | null>;
  /** Exception text if the run crashed. */
  error?: string | null;
  failure_category?: FailureCategory | null;
  failure_note?: string | null;
}

export interface RunFile {
  schema_version: number;
  run_id: string;
  /** ISO 8601 datetime. */
  created_at: string;
  git_sha?: string | null;
  config_name: string;
  config: Record<string, unknown>;
  split: Split;
  datasets: DatasetName[];
  summary: Record<string, MetricValue>;
  /** dataset -> metric -> value */
  summary_by_dataset: Record<string, Record<string, MetricValue>>;
  usage_total: Usage;
  results: QuestionResult[];
}

// ---------------------------------------------------------------- /api/chat

export interface ChatFilters {
  ticker?: string;
  fiscal_year?: number;
  form?: string;
}

export interface ChatRequest {
  question: string;
  filters?: ChatFilters;
  model?: string;
}

export interface StepEventData {
  /** plan, search_filings, get_company_facts, verify, answer, self_check, ... */
  name: string;
  detail: string;
}

export type ChatEvent =
  | { type: "step"; data: StepEventData }
  | { type: "token"; data: { text: string } }
  | { type: "answer"; data: AgentAnswer }
  | { type: "error"; data: { message: string } }
  | { type: "done"; data: Record<string, never> };

// ---------------------------------------------------------------- other endpoints

export interface ModelInfo {
  id: string;
  label: string;
  provider: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
  default: string;
}

export interface CompanyInfo {
  ticker: string;
  name: string;
  sector: string;
}

export interface CompaniesResponse {
  companies: CompanyInfo[];
}

// ---------------------------------------------------------------- dashboard.json

export interface DashboardRun {
  run_id: string;
  config_name: string;
  split: Split;
  created_at: string;
  datasets: string[];
  summary: Record<string, MetricValue>;
}

export interface TableCell {
  value: number;
  ci_low?: number | null;
  ci_high?: number | null;
  /** Difference vs the baseline row, in the metric's own units. */
  delta?: number | null;
  p_value?: number | null;
}

export interface TableRow {
  name: string;
  run_id: string;
  metrics: Record<string, TableCell>;
}

export interface ResultTable {
  /** e.g. ablation-retrieval, ablation-chunking, ablation-xbrl-verify, model-comparison, error-fixes */
  id: string;
  title: string;
  description: string;
  /** Name of the baseline row. */
  baseline: string;
  /** Metric names, first one is the main metric. */
  columns: string[];
  rows: TableRow[];
}

export interface FailureExample {
  qid: string;
  dataset: string;
  question: string;
  gold: string | null;
  answer: string | null;
  failure_category: string;
  run_id: string;
  explanation?: string | null;
}

/** Agreement of the LLM judge with human labels on a held-out split. */
export interface JudgeCalibration {
  judge_model: string;
  prompt_version: string;
  split: string;
  n: number;
  /** Three-way agreement over `labels`. */
  accuracy: MetricValue;
  kappa: MetricValue;
  error_rate: MetricValue;
  /** Agreement after collapsing to correct vs not correct. */
  binary_accuracy: MetricValue;
  binary_kappa: MetricValue;
  /** e.g. ["CORRECT", "INCORRECT", "REFUSAL"] */
  labels: string[];
  /** Rows are the human label, columns the judge label, both in `labels` order. */
  confusion: number[][];
  per_label_recall: Record<string, number>;
  /** Labels the judge output could not be parsed for. */
  parse_errors?: number | null;
  created_at?: string | null;
  /** The subset of examples never seen while tuning the judge prompt. */
  clean_n?: number | null;
  clean_accuracy?: MetricValue | null;
  clean_kappa?: MetricValue | null;
}

export interface Dashboard {
  generated_at: string;
  headline: {
    run_id: string;
    title: string;
    metrics: Record<string, MetricValue>;
  };
  runs: DashboardRun[];
  tables: ResultTable[];
  failure_counts: Record<string, number>;
  failures: FailureExample[];
  judge_calibration?: JudgeCalibration | null;
}
