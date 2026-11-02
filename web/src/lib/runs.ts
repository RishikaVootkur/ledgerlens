import type { QuestionResult } from "../api/types";
import { isRetrievalMetric, orderMetrics } from "./metrics";

export type Outcome = "correct" | "incorrect" | "error" | "unscored";

/** Score keys that decide correctness, in priority order. */
export const PRIMARY_SCORES = ["numeric_correct", "judge_correct", "refusal_correct"] as const;

export function primaryScore(result: QuestionResult): { name: string; value: number } | null {
  for (const name of PRIMARY_SCORES) {
    const value = result.scores[name];
    if (value !== null && value !== undefined) return { name, value };
  }
  return null;
}

export function questionOutcome(result: QuestionResult): Outcome {
  if (result.error) return "error";
  const score = primaryScore(result);
  if (!score) return "unscored";
  return score.value >= 0.5 ? "correct" : "incorrect";
}

/** True when no question has a prediction or a correctness score (a retrieval-only run). */
export function isRetrievalOnlyRun(results: QuestionResult[]): boolean {
  const scored = results.filter((r) => !r.error);
  return scored.length > 0 && scored.every((r) => r.prediction === null && primaryScore(r) === null);
}

/** Retrieval metric keys present in any question's scores, in display order. */
export function retrievalScoreNames(results: QuestionResult[]): string[] {
  const names = new Set<string>();
  for (const r of results) {
    for (const [k, v] of Object.entries(r.scores)) if (v != null && isRetrievalMetric(k)) names.add(k);
  }
  return orderMetrics([...names]);
}

export interface ResultFilters {
  dataset: string;
  outcome: Outcome | "all" | "refused";
  search: string;
}

export function filterResults(results: QuestionResult[], f: ResultFilters): QuestionResult[] {
  const needle = f.search.trim().toLowerCase();
  return results.filter((r) => {
    if (f.dataset !== "all" && r.dataset !== f.dataset) return false;
    if (f.outcome === "refused") {
      if (!r.prediction?.refused) return false;
    } else if (f.outcome !== "all" && questionOutcome(r) !== f.outcome) {
      return false;
    }
    if (needle && !`${r.qid} ${r.question} ${r.ticker ?? ""}`.toLowerCase().includes(needle)) {
      return false;
    }
    return true;
  });
}
