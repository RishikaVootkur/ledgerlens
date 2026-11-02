import { describe, expect, it } from "vitest";
import dashboardJson from "../../mock/dashboard.json";
import runJson from "../../mock/runs/2026-10-28T1402-full-gemini-flash.json";
import { FAILURE_CATEGORIES, SCHEMA_VERSION, type Dashboard, type MetricValue, type RunFile } from "./types";

// JSON imports widen literal unions to string, so check the shape at runtime
// and then treat the files as the declared types.
const dashboard = dashboardJson as unknown as Dashboard;
const run = runJson as unknown as RunFile;
const DATASETS = ["financebench", "xbrl", "unanswerable", "smoke"];

function expectMetricValue(m: MetricValue) {
  expect(typeof m.value).toBe("number");
  expect(Number.isInteger(m.n)).toBe(true);
  if (m.ci_low != null && m.ci_high != null) {
    expect(m.ci_low).toBeLessThanOrEqual(m.value);
    expect(m.ci_high).toBeGreaterThanOrEqual(m.value);
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
    const total = dashboard.judge_calibration!.confusion.flat().reduce((a, b) => a + b, 0);
    expect(total).toBe(dashboard.judge_calibration!.n);
  });
});

describe("mock run file", () => {
  it("matches the RunFile contract", () => {
    expect(run.schema_version).toBe(SCHEMA_VERSION);
    expect(Number.isNaN(Date.parse(run.created_at))).toBe(false);
    expect(dashboard.runs.some((r) => r.run_id === run.run_id)).toBe(true);
    Object.values(run.summary).forEach(expectMetricValue);
    Object.values(run.summary_by_dataset).forEach((m) => Object.values(m).forEach(expectMetricValue));
    expect(typeof run.usage_total.cost_usd).toBe("number");
    for (const r of run.results) {
      expect(DATASETS).toContain(r.dataset);
      expect(typeof r.answerable).toBe("boolean");
      if (r.failure_category != null) expect(FAILURE_CATEGORIES).toContain(r.failure_category);
      if (r.prediction) {
        expect(Array.isArray(r.prediction.citations)).toBe(true);
        expect(typeof r.prediction.usage.latency_ms).toBe("number");
      }
    }
  });
});
