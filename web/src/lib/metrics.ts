export type MetricFormat = "percent" | "ratio" | "ms" | "usd" | "count";

export interface MetricSpec {
  label: string;
  short?: string;
  format: MetricFormat;
  /** Latency and cost improve when they go down. */
  higherIsBetter: boolean;
  help?: string;
}

// Keys match the eval runner's run files (QuestionResult.scores and RunFile.summary).
export const METRICS: Record<string, MetricSpec> = {
  numeric_correct: {
    label: "Numeric accuracy",
    format: "percent",
    higherIsBetter: true,
    help: "Answer number within tolerance of the gold value",
  },
  judge_correct: {
    label: "Judge accuracy",
    format: "percent",
    higherIsBetter: true,
    help: "Correct according to the calibrated LLM judge",
  },
  citation_precision: {
    label: "Citation precision",
    format: "percent",
    higherIsBetter: true,
    help: "Share of citations that support the claim",
  },
  citation_recall: {
    label: "Citation recall",
    format: "percent",
    higherIsBetter: true,
    help: "Share of gold evidence that was cited",
  },
  refusal_correct: {
    label: "Refusal accuracy",
    format: "percent",
    higherIsBetter: true,
    help: "Refuses unanswerable questions and answers the rest",
  },
  recall_at_1: { label: "Recall@1", format: "percent", higherIsBetter: true },
  recall_at_5: { label: "Recall@5", format: "percent", higherIsBetter: true },
  recall_at_10: { label: "Recall@10", format: "percent", higherIsBetter: true },
  mrr: { label: "MRR", format: "ratio", higherIsBetter: true, help: "Mean reciprocal rank" },
  ndcg_at_10: { label: "nDCG@10", format: "ratio", higherIsBetter: true },
  model_latency_p50_ms: {
    label: "Model latency p50",
    format: "ms",
    higherIsBetter: false,
    help: "Time spent in model calls per question",
  },
  model_latency_p95_ms: {
    label: "Model latency p95",
    format: "ms",
    higherIsBetter: false,
    help: "Time spent in model calls per question",
  },
  latency_p50_ms: {
    label: "Wall latency p50",
    format: "ms",
    higherIsBetter: false,
    help: "End-to-end wall clock, including rate-limit waits",
  },
  latency_p95_ms: {
    label: "Wall latency p95",
    format: "ms",
    higherIsBetter: false,
    help: "End-to-end wall clock, including rate-limit waits",
  },
  tokens_mean: { label: "Tokens / question", format: "count", higherIsBetter: false },
  cost_usd_mean: {
    label: "Cost / question",
    short: "Cost / q",
    format: "usd",
    higherIsBetter: false,
    help: "List-price equivalent, even when the provider was free",
  },
  error_rate: {
    label: "Error rate",
    format: "percent",
    higherIsBetter: false,
    help: "Questions where the run crashed",
  },
  kappa: { label: "Judge agreement (κ)", format: "ratio", higherIsBetter: true },
};

/** Metrics that describe speed or spend; kept out of per-dataset quality tables. */
export function isCostMetric(name: string): boolean {
  return /latency|^tokens_|^cost_|^error_rate$/.test(name);
}

/** Retrieval-only metrics: present even when a run generated no answers. */
export function isRetrievalMetric(name: string): boolean {
  return /^recall_at_\d+$|^mrr$|^ndcg_at_\d+$/.test(name);
}

/** Axis domain for a value and its interval: 0..1 for rates, padded around the data otherwise. */
export function metricDomain(name: string, values: number[]): [number, number] {
  const spec = metricSpec(name);
  if (spec.format === "percent") return [0, 1];
  if (spec.format === "ratio" && values.every((v) => v >= 0 && v <= 1)) return [0, 1];
  if (values.length === 0) return [0, 1];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.35;
  return [Math.max(0, lo - pad), hi + pad];
}

/** Known metrics in display order, unknown ones after them alphabetically. */
export function orderMetrics(names: string[]): string[] {
  const order = Object.keys(METRICS);
  const rank = (n: string) => {
    const i = order.indexOf(n);
    return i === -1 ? order.length : i;
  };
  return [...names].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

function humanize(name: string): string {
  const words = name.replace(/_/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function metricSpec(name: string): MetricSpec {
  return METRICS[name] ?? { label: humanize(name), format: "ratio", higherIsBetter: true };
}

export function metricLabel(name: string): string {
  return metricSpec(name).label;
}

const intFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** Format a metric value for display, e.g. 0.712 -> "71.2%", 6400 -> "6.40 s". */
export function formatMetric(name: string, value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  switch (metricSpec(name).format) {
    case "percent":
      return `${(value * 100).toFixed(1)}%`;
    case "ratio":
      return value.toFixed(3);
    case "ms":
      return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
    case "usd":
      return formatUSD(value);
    case "count":
      return intFmt.format(value);
  }
}

export function formatUSD(value: number): string {
  if (value === 0) return "$0";
  if (Math.abs(value) >= 1) return `$${value.toFixed(2)}`;
  // Small per-question costs: keep 2 significant digits.
  return `$${Number(value.toPrecision(2)).toString()}`;
}

/** "[67.1%, 75.4%]", or "" when the interval is missing. */
export function formatCI(
  name: string,
  low: number | null | undefined,
  high: number | null | undefined,
): string {
  if (low === null || low === undefined || high === null || high === undefined) return "";
  return `[${formatMetric(name, low)}, ${formatMetric(name, high)}]`;
}

/** Signed difference in the metric's units: percent metrics use percentage points. */
export function formatDelta(name: string, delta: number | null | undefined): string {
  if (delta === null || delta === undefined || Number.isNaN(delta)) return "";
  const sign = delta > 0 ? "+" : delta < 0 ? "−" : "±";
  const abs = Math.abs(delta);
  switch (metricSpec(name).format) {
    case "percent":
      return `${sign}${(abs * 100).toFixed(1)} pts`;
    case "ratio":
      return `${sign}${abs.toFixed(3)}`;
    case "ms":
      return abs >= 1000 ? `${sign}${(abs / 1000).toFixed(2)} s` : `${sign}${Math.round(abs)} ms`;
    case "usd":
      return `${sign}${formatUSD(abs)}`;
    case "count":
      return `${sign}${intFmt.format(abs)}`;
  }
}

export type DeltaTone = "better" | "worse" | "neutral";

/** Whether a delta is an improvement, taking the metric's direction into account. */
export function deltaTone(name: string, delta: number | null | undefined): DeltaTone {
  if (delta === null || delta === undefined || delta === 0 || Number.isNaN(delta)) return "neutral";
  const up = delta > 0;
  return up === metricSpec(name).higherIsBetter ? "better" : "worse";
}

export function isSignificant(p: number | null | undefined, alpha = 0.05): boolean {
  return p !== null && p !== undefined && p < alpha;
}

export function formatPValue(p: number | null | undefined): string {
  if (p === null || p === undefined) return "";
  return p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`;
}
