import type { QuestionResult } from "../api/types";

export type Outcome = "correct" | "incorrect" | "error" | "unscored";

/** Score keys that decide correctness, in priority order. */
export const PRIMARY_SCORES = ["numeric_accuracy", "judge_accuracy", "refusal_accuracy"] as const;

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
