import type { Outcome } from "../../lib/runs";
import { Badge } from "../ui";

const OUTCOME: Record<Outcome, { label: string; tone: "good" | "bad" | "warn" | "neutral" }> = {
  correct: { label: "Correct", tone: "good" },
  incorrect: { label: "Incorrect", tone: "bad" },
  error: { label: "Error", tone: "warn" },
  unscored: { label: "Unscored", tone: "neutral" },
};

export function OutcomeBadge({ outcome }: { outcome: Outcome }) {
  const o = OUTCOME[outcome];
  return <Badge tone={o.tone}>{o.label}</Badge>;
}
