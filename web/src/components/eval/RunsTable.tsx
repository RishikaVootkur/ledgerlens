import { useMemo, useState } from "react";
import type { DashboardRun } from "../../api/types";
import { formatMetric, metricLabel } from "../../lib/metrics";
import { href } from "../../lib/router";
import { Badge, Card, SectionHeader } from "../ui";
import { formatDate } from "../../lib/style";

const SUMMARY_COLUMNS = ["numeric_accuracy", "citation_precision", "latency_p50_ms"];

export function RunsTable({ runs }: { runs: DashboardRun[] }) {
  const [newestFirst, setNewestFirst] = useState(true);
  const sorted = useMemo(
    () =>
      [...runs].sort((a, b) => {
        const diff = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        return newestFirst ? -diff : diff;
      }),
    [runs, newestFirst],
  );
  return (
    <Card className="overflow-hidden" labelledBy="runs-heading">
      <div className="p-4 sm:p-5">
        <SectionHeader
          id="runs-heading"
          title="Runs"
          description="Every number on this page comes from one of these run files."
        />
      </div>
      <div className="overflow-x-auto border-t border-zinc-200 dark:border-zinc-800">
        <table className="w-full text-sm">
          <thead className="bg-zinc-50 text-xs text-zinc-500 dark:bg-zinc-900/60 dark:text-zinc-400">
            <tr>
              <th scope="col" className="px-4 py-2 text-left font-medium sm:pl-5">
                Config
              </th>
              <th scope="col" className="px-4 py-2 text-left font-medium">
                Split
              </th>
              <th
                scope="col"
                className="px-4 py-2 text-left font-medium"
                aria-sort={newestFirst ? "descending" : "ascending"}
              >
                <button
                  type="button"
                  onClick={() => setNewestFirst((v) => !v)}
                  className="inline-flex items-center gap-1 rounded hover:text-zinc-900 dark:hover:text-zinc-100"
                >
                  Created
                  <span aria-hidden="true">{newestFirst ? "↓" : "↑"}</span>
                  <span className="sr-only">, sort {newestFirst ? "oldest" : "newest"} first</span>
                </button>
              </th>
              {SUMMARY_COLUMNS.map((c) => (
                <th key={c} scope="col" className="px-4 py-2 text-right font-medium whitespace-nowrap">
                  {metricLabel(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
            {sorted.map((run) => (
              <tr key={run.run_id} className="hover:bg-zinc-50/70 dark:hover:bg-zinc-800/30">
                <th scope="row" className="px-4 py-2.5 text-left font-normal sm:pl-5">
                  <a
                    href={href.run(run.run_id)}
                    className="font-medium text-zinc-900 hover:text-accent-700 hover:underline dark:text-zinc-100 dark:hover:text-accent-300"
                  >
                    {run.config_name}
                  </a>
                  <span className="block font-mono text-[11px] text-zinc-500 dark:text-zinc-400">{run.run_id}</span>
                </th>
                <td className="px-4 py-2.5">
                  <Badge tone={run.split === "test" ? "accent" : "neutral"}>{run.split}</Badge>
                </td>
                <td className="px-4 py-2.5 whitespace-nowrap text-zinc-600 tabular-nums dark:text-zinc-400">
                  {formatDate(run.created_at)}
                </td>
                {SUMMARY_COLUMNS.map((c) => (
                  <td key={c} className="px-4 py-2.5 text-right tabular-nums">
                    {formatMetric(c, run.summary[c]?.value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
