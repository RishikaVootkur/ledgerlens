import { useCallback, useId, useMemo, useState } from "react";
import { getRun } from "../api/client";
import type { RunFile } from "../api/types";
import { Drawer } from "../components/Drawer";
import { MetricCard } from "../components/eval/MetricCard";
import { OutcomeBadge } from "../components/run/OutcomeBadge";
import { QuestionDetail } from "../components/run/QuestionDetail";
import { Badge, Card, ErrorNote, LoadingBlock, SectionHeader } from "../components/ui";
import { formatDate, inputClass } from "../lib/style";
import { failureLabel } from "../lib/failures";
import { formatMetric, formatUSD, isCostMetric, metricLabel, orderMetrics } from "../lib/metrics";
import { href, replaceHash } from "../lib/router";
import {
  filterResults,
  isRetrievalOnlyRun,
  primaryScore,
  questionOutcome,
  retrievalScoreNames,
  type ResultFilters,
} from "../lib/runs";
import { useAsync } from "../lib/useAsync";

export function RunPage({ runId, qid }: { runId: string; qid: string | null }) {
  const state = useAsync((signal) => getRun(runId, signal), runId);
  if (state.status === "loading") return <LoadingBlock label="Loading run" />;
  if (state.status === "error") {
    return (
      <div className="space-y-4">
        <BackLink />
        <ErrorNote title={`Could not load run ${runId}`} error={state.error} />
      </div>
    );
  }
  return <RunView run={state.data} qid={qid} />;
}

function BackLink() {
  return (
    <a href={href.eval()} className="text-sm text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100">
      ← Evaluation lab
    </a>
  );
}

const OUTCOMES: Array<{ value: ResultFilters["outcome"]; label: string }> = [
  { value: "all", label: "All" },
  { value: "correct", label: "Correct" },
  { value: "incorrect", label: "Incorrect" },
  { value: "refused", label: "Refused" },
  { value: "error", label: "Errored" },
  { value: "unscored", label: "Unscored" },
];

function RunView({ run, qid }: { run: RunFile; qid: string | null }) {
  const ids = { ds: useId(), out: useId(), q: useId() };
  const [filters, setFilters] = useState<ResultFilters>({ dataset: "all", outcome: "all", search: "" });
  const rows = useMemo(() => filterResults(run.results, filters), [run.results, filters]);
  const selected = qid ? run.results.find((r) => r.qid === qid) : undefined;
  const summaryNames = orderMetrics(Object.keys(run.summary));
  const datasets = Object.keys(run.summary_by_dataset ?? {});
  const datasetMetrics = orderMetrics([
    ...new Set(datasets.flatMap((d) => Object.keys(run.summary_by_dataset[d] ?? {}))),
  ]).filter((m) => !isCostMetric(m));
  // Retrieval-only runs have no predictions: show retrieval scores instead of outcomes.
  const retrievalOnly = useMemo(() => isRetrievalOnlyRun(run.results), [run.results]);
  const retrievalColumns = useMemo(
    () => (retrievalOnly ? retrievalScoreNames(run.results) : []),
    [retrievalOnly, run.results],
  );
  const columnCount = 2 + (retrievalOnly ? retrievalColumns.length : 3);

  const open = (id: string) => replaceHash(href.run(run.run_id, id));
  const close = useCallback(() => replaceHash(href.run(run.run_id)), [run.run_id]);

  const counts = useMemo(() => {
    const c = { correct: 0, incorrect: 0, error: 0, unscored: 0 };
    for (const r of run.results) c[questionOutcome(r)]++;
    return c;
  }, [run.results]);

  return (
    <div className="space-y-6">
      <BackLink />
      <header className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold tracking-tight">{run.config_name}</h1>
          <Badge tone={run.split === "test" ? "accent" : "neutral"}>{run.split}</Badge>
        </div>
        <p className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-500 tabular-nums dark:text-zinc-400">
          <span className="font-mono">{run.run_id}</span>
          <span>{formatDate(run.created_at)}</span>
          {run.git_sha ? <span className="font-mono">git {run.git_sha.slice(0, 7)}</span> : null}
          <span>{run.datasets.join(", ")}</span>
          {retrievalOnly ? (
            <span>
              {run.results.length} questions · retrieval only
              {counts.error ? ` · ${counts.error} errored` : ""}
            </span>
          ) : (
            <span>
              {run.results.length} questions · {counts.correct} correct · {counts.incorrect} incorrect
              {counts.error ? ` · ${counts.error} errored` : ""}
              {counts.unscored ? ` · ${counts.unscored} unscored` : ""}
            </span>
          )}
          {run.usage_total && run.usage_total.llm_calls > 0 ? (
            <span title="List-price equivalent">
              {formatUSD(run.usage_total.cost_usd)} total · {run.usage_total.llm_calls} LLM calls
            </span>
          ) : null}
        </p>
      </header>

      <section aria-label="Run summary" className="grid grid-cols-1 gap-3 min-[420px]:grid-cols-2 lg:grid-cols-4">
        {summaryNames.map((name) => (
          <MetricCard key={name} name={name} metric={run.summary[name]!} />
        ))}
      </section>

      {datasets.length > 0 && datasetMetrics.length > 0 ? (
        <Card className="overflow-hidden" labelledBy="by-dataset">
          <div className="p-4 sm:p-5">
            <SectionHeader id="by-dataset" title="By dataset" />
          </div>
          <div className="relative overflow-x-auto border-t border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-sm tabular-nums">
              <thead className="bg-zinc-50 text-xs text-zinc-500 dark:bg-zinc-900/60 dark:text-zinc-400">
                <tr>
                  <th scope="col" className="px-4 py-2 text-left font-medium sm:pl-5">
                    Dataset
                  </th>
                  {datasetMetrics.map((m) => (
                    <th key={m} scope="col" className="px-4 py-2 text-right font-medium whitespace-nowrap">
                      {metricLabel(m)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                {datasets.map((d) => (
                  <tr key={d}>
                    <th scope="row" className="px-4 py-2 text-left font-medium sm:pl-5">
                      {d}
                      <span className="ml-1.5 text-xs font-normal text-zinc-500 dark:text-zinc-400">
                        n={Math.max(0, ...Object.values(run.summary_by_dataset[d] ?? {}).map((v) => v.n))}
                      </span>
                    </th>
                    {datasetMetrics.map((m) => {
                      const v = run.summary_by_dataset[d]?.[m];
                      return (
                        <td key={m} className="px-4 py-2 text-right">
                          {v ? formatMetric(m, v.value) : <span className="text-zinc-500 dark:text-zinc-400">n/a</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : null}

      <Card className="overflow-hidden" labelledBy="results-heading">
        <div className="space-y-4 p-4 sm:p-5">
          <SectionHeader
            id="results-heading"
            title="Per-question results"
            aside={
              <span className="text-xs text-zinc-500 tabular-nums dark:text-zinc-400">
                {rows.length} of {run.results.length}
              </span>
            }
          />
          <div className="grid gap-3 sm:grid-cols-[10rem_10rem_minmax(0,1fr)]">
            <div>
              <label htmlFor={ids.ds} className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                Dataset
              </label>
              <select
                id={ids.ds}
                className={inputClass}
                value={filters.dataset}
                onChange={(e) => setFilters((f) => ({ ...f, dataset: e.target.value }))}
              >
                <option value="all">All datasets</option>
                {run.datasets.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
            <div className={retrievalOnly ? "hidden" : undefined}>
              <label htmlFor={ids.out} className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                Outcome
              </label>
              <select
                id={ids.out}
                className={inputClass}
                value={filters.outcome}
                onChange={(e) =>
                  setFilters((f) => ({ ...f, outcome: e.target.value as ResultFilters["outcome"] }))
                }
              >
                {OUTCOMES.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor={ids.q} className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                Search
              </label>
              <input
                id={ids.q}
                type="search"
                className={inputClass}
                placeholder="Question, ticker or id"
                value={filters.search}
                onChange={(e) => setFilters((f) => ({ ...f, search: e.target.value }))}
              />
            </div>
          </div>
        </div>
        <div className="relative overflow-x-auto border-t border-zinc-200 dark:border-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-zinc-50 text-xs text-zinc-500 dark:bg-zinc-900/60 dark:text-zinc-400">
              <tr>
                <th scope="col" className="px-4 py-2 text-left font-medium sm:w-2/5 sm:pl-5">
                  Question
                </th>
                <th scope="col" className="hidden px-4 py-2 text-left font-medium md:table-cell">
                  Dataset
                </th>
                {retrievalOnly ? (
                  retrievalColumns.map((m) => (
                    <th key={m} scope="col" className="px-4 py-2 text-right font-medium whitespace-nowrap">
                      {metricLabel(m)}
                    </th>
                  ))
                ) : (
                  <>
                    <th scope="col" className="px-4 py-2 text-left font-medium">
                      Outcome
                    </th>
                    <th scope="col" className="hidden px-4 py-2 text-right font-medium sm:table-cell">
                      Score
                    </th>
                    <th scope="col" className="hidden px-4 py-2 text-left font-medium lg:table-cell">
                      Failure
                    </th>
                  </>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {rows.map((r) => {
                const score = primaryScore(r);
                return (
                  <tr key={r.qid} className="hover:bg-zinc-50/70 dark:hover:bg-zinc-800/30">
                    <td className="max-w-0 px-4 py-2.5 sm:pl-5">
                      <button
                        type="button"
                        onClick={() => open(r.qid)}
                        className="block w-full truncate rounded text-left text-zinc-900 hover:text-accent-700 dark:text-zinc-100 dark:hover:text-accent-300"
                        title={r.question}
                      >
                        {r.question}
                      </button>
                      <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400">{r.qid}</span>
                    </td>
                    <td className="hidden px-4 py-2.5 text-zinc-600 md:table-cell dark:text-zinc-400">{r.dataset}</td>
                    {retrievalOnly ? (
                      retrievalColumns.map((m) => (
                        <td key={m} className="px-4 py-2.5 text-right whitespace-nowrap tabular-nums">
                          {r.scores[m] != null ? (
                            formatMetric(m, r.scores[m])
                          ) : (
                            <span className="text-zinc-500 dark:text-zinc-400">n/a</span>
                          )}
                        </td>
                      ))
                    ) : (
                      <>
                        <td className="px-4 py-2.5">
                          <OutcomeBadge outcome={questionOutcome(r)} />
                        </td>
                        <td className="hidden px-4 py-2.5 text-right whitespace-nowrap tabular-nums sm:table-cell">
                          {score ? (
                            <span title={metricLabel(score.name)}>{formatMetric(score.name, score.value)}</span>
                          ) : (
                            <span className="text-zinc-500 dark:text-zinc-400">n/a</span>
                          )}
                        </td>
                        <td className="hidden px-4 py-2.5 text-zinc-600 lg:table-cell dark:text-zinc-400">
                          {r.failure_category ? failureLabel(r.failure_category) : ""}
                        </td>
                      </>
                    )}
                  </tr>
                );
              })}
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={columnCount}className="px-5 py-6 text-center text-sm text-zinc-500 dark:text-zinc-400">
                    No questions match these filters.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </Card>

      <Drawer open={selected !== undefined} onClose={close} title={selected ? `Question ${selected.qid}` : ""}>
        {selected ? <QuestionDetail result={selected} /> : null}
      </Drawer>
    </div>
  );
}
