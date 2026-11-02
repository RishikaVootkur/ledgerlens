import { describe, expect, it } from "vitest";
import {
  deltaTone,
  formatCI,
  formatDelta,
  formatMetric,
  formatPValue,
  isSignificant,
  metricLabel,
  orderMetrics,
} from "./metrics";

describe("formatMetric", () => {
  it("formats each metric family", () => {
    expect(formatMetric("numeric_correct", 0.7124)).toBe("71.2%");
    expect(formatMetric("recall_at_10", 1)).toBe("100.0%");
    expect(formatMetric("mrr", 0.61234)).toBe("0.612");
    expect(formatMetric("latency_p50_ms", 6420)).toBe("6.42 s");
    expect(formatMetric("latency_p95_ms", 850.4)).toBe("850 ms");
    expect(formatMetric("tokens_mean", 9850)).toBe("9,850");
    expect(formatMetric("cost_usd_mean", 0.0041)).toBe("$0.0041");
    expect(formatMetric("cost_usd_mean", 0.004123)).toBe("$0.0041");
    expect(formatMetric("cost_usd_mean", 1.4762)).toBe("$1.48");
  });

  it("handles missing values and unknown metrics", () => {
    expect(formatMetric("numeric_correct", null)).toBe("n/a");
    expect(formatMetric("numeric_correct", undefined)).toBe("n/a");
    expect(formatMetric("some_new_metric", 0.5)).toBe("0.500");
    expect(metricLabel("some_new_metric")).toBe("Some new metric");
    expect(metricLabel("cost_usd_mean")).toBe("Cost / question");
  });
});

describe("intervals and deltas", () => {
  it("formats confidence intervals", () => {
    expect(formatCI("numeric_correct", 0.666, 0.758)).toBe("[66.6%, 75.8%]");
    expect(formatCI("numeric_correct", null, 0.7)).toBe("");
  });

  it("formats deltas in the metric's units", () => {
    expect(formatDelta("numeric_correct", 0.032)).toBe("+3.2 pts");
    expect(formatDelta("numeric_correct", -0.081)).toBe("−8.1 pts");
    expect(formatDelta("latency_p50_ms", -3150)).toBe("−3.15 s");
    expect(formatDelta("cost_usd_mean", 0.0021)).toBe("+$0.0021");
    expect(formatDelta("mrr", 0)).toBe("±0.000");
    expect(formatDelta("mrr", null)).toBe("");
  });

  it("colors by improvement, respecting lower-is-better metrics", () => {
    expect(deltaTone("numeric_correct", 0.02)).toBe("better");
    expect(deltaTone("numeric_correct", -0.02)).toBe("worse");
    expect(deltaTone("latency_p50_ms", -500)).toBe("better");
    expect(deltaTone("cost_usd_mean", 0.001)).toBe("worse");
    expect(deltaTone("mrr", 0)).toBe("neutral");
  });

  it("flags significance at p < 0.05", () => {
    expect(isSignificant(0.049)).toBe(true);
    expect(isSignificant(0.05)).toBe(false);
    expect(isSignificant(undefined)).toBe(false);
    expect(formatPValue(0.0001)).toBe("p < 0.001");
    expect(formatPValue(0.034)).toBe("p = 0.034");
  });
});

describe("run file metric names", () => {
  it("labels every key the eval runner writes", () => {
    const labels = {
      numeric_correct: "Numeric accuracy",
      judge_correct: "Judge accuracy",
      refusal_correct: "Refusal accuracy",
      recall_at_1: "Recall@1",
      recall_at_5: "Recall@5",
      recall_at_10: "Recall@10",
      ndcg_at_10: "nDCG@10",
      latency_p50_ms: "Wall latency p50",
      model_latency_p95_ms: "Model latency p95",
      tokens_mean: "Tokens / question",
      cost_usd_mean: "Cost / question",
      error_rate: "Error rate",
    };
    for (const [key, label] of Object.entries(labels)) expect(metricLabel(key)).toBe(label);
    expect(formatMetric("ndcg_at_10", 0.9077)).toBe("0.908");
    expect(formatMetric("error_rate", 0.05)).toBe("5.0%");
  });

  it("treats error rate and model latency as lower-is-better", () => {
    expect(deltaTone("error_rate", 0.01)).toBe("worse");
    expect(deltaTone("model_latency_p50_ms", -100)).toBe("better");
  });
});

describe("orderMetrics", () => {
  it("puts known metrics first in display order", () => {
    expect(orderMetrics(["zeta", "latency_p50_ms", "numeric_correct", "alpha"])).toEqual([
      "numeric_correct",
      "latency_p50_ms",
      "alpha",
      "zeta",
    ]);
  });
});
