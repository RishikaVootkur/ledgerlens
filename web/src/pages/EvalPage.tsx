import { getDashboard } from "../api/client";
import { FailureTaxonomy } from "../components/eval/FailureTaxonomy";
import { JudgeCalibrationCard } from "../components/eval/JudgeCalibrationCard";
import { MetricCard } from "../components/eval/MetricCard";
import { ResultTableView } from "../components/eval/ResultTableView";
import { RunsTable } from "../components/eval/RunsTable";
import { ErrorNote, LoadingBlock } from "../components/ui";
import { formatDate } from "../lib/style";
import { DATA_BASE } from "../lib/env";
import { orderMetrics } from "../lib/metrics";
import { href } from "../lib/router";
import { useAsync } from "../lib/useAsync";

export function EvalPage() {
  const state = useAsync((signal) => getDashboard(signal), "dashboard");
  if (state.status === "loading") return <LoadingBlock label="Loading eval results" />;
  if (state.status === "error") {
    return (
      <ErrorNote
        title={`Could not load ${DATA_BASE}/dashboard.json`}
        error={`${state.error.message}. Build it with the eval CLI, or run the UI with VITE_DATA_BASE=/mock for sample data.`}
      />
    );
  }
  const d = state.data;
  const headlineNames = orderMetrics(Object.keys(d.headline.metrics));

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Evaluation lab</h1>
          <p className="mt-1 max-w-3xl text-sm text-zinc-600 dark:text-zinc-400">
            How often LedgerLens gets the number right and cites real support, with 95% bootstrap
            confidence intervals and paired significance tests for every comparison.
          </p>
        </div>
        <p className="text-xs text-zinc-500 tabular-nums dark:text-zinc-400">Updated {formatDate(d.generated_at)}</p>
      </header>

      <section aria-labelledby="headline-heading" className="space-y-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="headline-heading" className="text-base font-semibold tracking-tight">
            {d.headline.title}
          </h2>
          <a
            href={href.run(d.headline.run_id)}
            className="text-sm font-medium text-accent-700 hover:underline dark:text-accent-300"
          >
            Per-question results →
          </a>
        </div>
        <div className="grid grid-cols-1 gap-3 min-[420px]:grid-cols-2 lg:grid-cols-4">
          {headlineNames.map((name) => (
            <MetricCard key={name} name={name} metric={d.headline.metrics[name]!} />
          ))}
          {d.judge_calibration && !("kappa" in d.headline.metrics) ? (
            <MetricCard name="kappa" metric={d.judge_calibration.kappa} />
          ) : null}
        </div>
      </section>

      {/* Wide tables (many columns) span the page; the rest flow into two balanced columns. */}
      {d.tables
        .filter((t) => t.columns.length > 4)
        .map((t) => (
          <ResultTableView key={t.id} table={t} />
        ))}
      <div className="gap-6 xl:columns-2">
        {d.tables
          .filter((t) => t.columns.length <= 4)
          .map((t) => (
            <div key={t.id} className="mb-6 break-inside-avoid">
              <ResultTableView table={t} />
            </div>
          ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <FailureTaxonomy counts={d.failure_counts} failures={d.failures} />
        {d.judge_calibration ? <JudgeCalibrationCard calibration={d.judge_calibration} /> : null}
      </div>

      <RunsTable runs={d.runs} />

      <p className="text-xs text-zinc-500 dark:text-zinc-400">
        Deltas compare each variant with the table's baseline. Green is an improvement, red a
        regression (lower is better for latency and cost). Bold deltas are significant at p &lt; 0.05;
        muted ones are not. Costs are list-price equivalents, even when the provider was free.
      </p>
    </div>
  );
}
