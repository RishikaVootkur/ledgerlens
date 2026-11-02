export const FAILURE_LABELS: Record<string, string> = {
  retrieval_miss: "Retrieval miss",
  table_misread: "Table misread",
  arithmetic: "Arithmetic",
  wrong_period: "Wrong period",
  hallucinated_citation: "Hallucinated citation",
  bad_refusal: "Bad refusal",
  other: "Other",
};

export function failureLabel(category: string): string {
  return FAILURE_LABELS[category] ?? category.replace(/_/g, " ");
}
