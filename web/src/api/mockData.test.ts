import { describe, expect, it } from "vitest";
import dashboardJson from "../../mock/dashboard.json";
import retrievalJson from "../../mock/runs/2026-10-27T0910-bm25.json";
import runJson from "../../mock/runs/2026-10-28T1402-full-gemini-flash.json";
import ciSliceJson from "../../mock/runs/ci-slice-baseline.json";
import { METRICS } from "../lib/metrics";
import { isRetrievalOnlyRun, questionOutcome } from "../lib/runs";
import { FAILURE_CATEGORIES, SCHEMA_VERSION, type Dashboard, type MetricValue, type RunFile } from "./types";

// JSON imports widen literal unions to string, so check the shape at runtime
// and then treat the files as the declared types.
const dashboard = dashboardJson as unknown as Dashboard;
const run = runJson as unknown as RunFile;
const retrievalRun = retrievalJson as unknown as RunFile;
// A real run file written by the eval runner (CI replay slice).
const ciSlice = ciSliceJson as unknown as RunFile;
const DATASETS = ["financebench", "xbrl", "unanswerable", "smoke"];

function expectMetricValue(m: MetricValue) {
  expect(typeof m.value).toBe("number");
  expect(Number.isInteger(m.n)).toBe(true);
  if (m.ci_low != null && m.ci_high != null) {
    expect(m.ci_low).toBeLessThanOrEqual(m.value);
    expect(m.ci_high).toBeGreaterThanOrEqual(m.value);
  }
}

function expectRunFile(r: RunFile) {
  expect(r.schema_version).toBe(SCHEMA_VERSION);
  expect(Number.isNaN(Date.parse(r.created_at))).toBe(false);
  Object.values(r.summary).forEach(expectMetricValue);
  Object.values(r.summary_by_dataset).forEach((m) => Object.values(m).forEach(expectMetricValue));
  expect(typeof r.usage_total.cost_usd).toBe("number");
  for (const q of r.results) {
    expect(DATASETS).toContain(q.dataset);
    expect(typeof q.answerable).toBe("boolean");
    if (q.failure_category != null) expect(FAILURE_CATEGORIES).toContain(q.failure_category);
    if (q.prediction) {
      expect(Array.isArray(q.prediction.citations)).toBe(true);
      expect(typeof q.prediction.usage.latency_ms).toBe("number");
    }
  }
}

describe("mock dashboard.json", () => {
  it("has every table cell, baseline and run reference", () => {
    const runIds = new Set(dashboard.runs.map((r) => r.run_id));
    expect(runIds.has(dashboard.headline.run_id)).toBe(true);
    Object.values(dashboard.headline.metrics).forEach(expectMetricValue);
    for (const t of dashboard.tables) {
      expect(t.rows.some((r) => r.name === t.baseline)).toBe(true);
      for (const row of t.rows) {
        expect(runIds.has(row.run_id)).toBe(true);
        for (const c of t.columns) expect(typeof row.metrics[c]?.value).toBe("number");
      }
    }
    for (const f of dashboard.failures) {
      expect(FAILURE_CATEGORIES).toContain(f.failure_category);
      expect(runIds.has(f.run_id)).toBe(true);
    }
  });

  it("has a three-label judge calibration whose confusion matrix sums to n", () => {
    const cal = dashboard.judge_calibration!;
    expect(cal.labels).toEqual(["CORRECT", "INCORRECT", "REFUSAL"]);
    expect(cal.confusion).toHaveLength(cal.labels.length);
    for (const row of cal.confusion) expect(row).toHaveLength(cal.labels.length);
    expect(cal.confusion.flat().reduce((a, b) => a + b, 0)).toBe(cal.n);
    expect(Object.keys(cal.per_label_recall).sort()).toEqual([...cal.labels].sort());
    [cal.accuracy, cal.kappa, cal.error_rate, cal.binary_accuracy, cal.binary_kappa].forEach(expectMetricValue);
  });

  it("only uses metric names the UI knows", () => {
    const names = new Set<string>();
    Object.keys(dashboard.headline.metrics).forEach((n) => names.add(n));
    dashboard.tables.forEach((t) => t.columns.forEach((c) => names.add(c)));
    dashboard.runs.forEach((r) => Object.keys(r.summary).forEach((n) => names.add(n)));
    for (const n of names) expect(METRICS, n).toHaveProperty([n]);
  });
});

describe("mock run files", () => {
  it("match the RunFile contract", () => {
    expectRunFile(run);
    expect(dashboard.runs.some((r) => r.run_id === run.run_id)).toBe(true);
    expect(isRetrievalOnlyRun(run.results)).toBe(false);
  });

  it("include a retrieval-only run with no predictions", () => {
    expectRunFile(retrievalRun);
    expect(retrievalRun.results.every((r) => r.prediction === null)).toBe(true);
    expect(isRetrievalOnlyRun(retrievalRun.results)).toBe(true);
  });

  it("read the real CI-slice run file with known metric names", () => {
    expectRunFile(ciSlice);
    for (const n of Object.keys(ciSlice.summary)) expect(METRICS, n).toHaveProperty([n]);
    for (const q of ciSlice.results) {
      for (const n of Object.keys(q.scores)) expect(METRICS, n).toHaveProperty([n]);
    }
    expect(ciSlice.results.filter((q) => questionOutcome(q) === "correct").length).toBeGreaterThan(0);
  });
});
