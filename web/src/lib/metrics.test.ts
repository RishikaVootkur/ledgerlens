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
    expect(formatMetric("numeric_accuracy", 0.7124)).toBe("71.2%");
    expect(formatMetric("recall@10", 1)).toBe("100.0%");
    expect(formatMetric("mrr", 0.61234)).toBe("0.612");
    expect(formatMetric("latency_p50_ms", 6420)).toBe("6.42 s");
    expect(formatMetric("latency_p95_ms", 850.4)).toBe("850 ms");
    expect(formatMetric("tokens_per_question", 9850)).toBe("9,850");
    expect(formatMetric("cost_per_question_usd", 0.0041)).toBe("$0.0041");
    expect(formatMetric("cost_per_question_usd", 0.004123)).toBe("$0.0041");
    expect(formatMetric("cost_per_question_usd", 1.4762)).toBe("$1.48");
  });

  it("handles missing values and unknown metrics", () => {
    expect(formatMetric("numeric_accuracy", null)).toBe("n/a");
    expect(formatMetric("numeric_accuracy", undefined)).toBe("n/a");
    expect(formatMetric("some_new_metric", 0.5)).toBe("0.500");
    expect(metricLabel("some_new_metric")).toBe("Some new metric");
    expect(metricLabel("cost_per_question_usd")).toBe("Cost / question");
  });
});

describe("intervals and deltas", () => {
  it("formats confidence intervals", () => {
    expect(formatCI("numeric_accuracy", 0.666, 0.758)).toBe("[66.6%, 75.8%]");
    expect(formatCI("numeric_accuracy", null, 0.7)).toBe("");
  });

  it("formats deltas in the metric's units", () => {
    expect(formatDelta("numeric_accuracy", 0.032)).toBe("+3.2 pts");
    expect(formatDelta("numeric_accuracy", -0.081)).toBe("−8.1 pts");
    expect(formatDelta("latency_p50_ms", -3150)).toBe("−3.15 s");
    expect(formatDelta("cost_per_question_usd", 0.0021)).toBe("+$0.0021");
    expect(formatDelta("mrr", 0)).toBe("±0.000");
    expect(formatDelta("mrr", null)).toBe("");
  });

  it("colors by improvement, respecting lower-is-better metrics", () => {
    expect(deltaTone("numeric_accuracy", 0.02)).toBe("better");
    expect(deltaTone("numeric_accuracy", -0.02)).toBe("worse");
    expect(deltaTone("latency_p50_ms", -500)).toBe("better");
    expect(deltaTone("cost_per_question_usd", 0.001)).toBe("worse");
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

describe("orderMetrics", () => {
  it("puts known metrics first in display order", () => {
    expect(orderMetrics(["zeta", "latency_p50_ms", "numeric_accuracy", "alpha"])).toEqual([
      "numeric_accuracy",
      "latency_p50_ms",
      "alpha",
      "zeta",
    ]);
  });
});
