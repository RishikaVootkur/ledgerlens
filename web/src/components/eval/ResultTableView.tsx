import type { ResultTable, TableCell } from "../../api/types";
import {
  deltaTone,
  formatCI,
  formatDelta,
  formatMetric,
  formatPValue,
  isSignificant,
  metricDomain,
  metricLabel,
  metricSpec,
} from "../../lib/metrics";
import { href } from "../../lib/router";
import { Badge, Card, SectionHeader } from "../ui";
import { cx } from "../../lib/style";
import { CIBar } from "./MetricCard";

const toneClass = {
  better: "text-emerald-700 dark:text-emerald-400",
  worse: "text-rose-700 dark:text-rose-400",
  neutral: "text-zinc-500 dark:text-zinc-400",
};

function DeltaText({ metric, cell }: { metric: string; cell: TableCell }) {
  if (cell.delta == null) return null;
  const sig = isSignificant(cell.p_value);
  const tone = deltaTone(metric, cell.delta);
  const p = formatPValue(cell.p_value);
  return (
    <span
      className={cx(
        "text-xs tabular-nums",
        sig ? cx("font-semibold", toneClass[tone]) : "text-zinc-500 dark:text-zinc-400",
      )}
      title={p ? `${p}${sig ? " (significant)" : " (not significant)"}` : undefined}
    >
      {formatDelta(metric, cell.delta)}
      {p ? <span className="sr-only">, {p}</span> : null}
    </span>
  );
}

function WhiskerChart({ table }: { table: ResultTable }) {
  const metric = table.columns[0];
  if (!metric) return null;
  const cells = table.rows
    .map((r) => ({ row: r, cell: r.metrics[metric] }))
    .filter((x): x is { row: (typeof table.rows)[number]; cell: TableCell } => x.cell !== undefined);
  if (cells.length === 0) return null;
  const values = cells.flatMap(({ cell }) =>
    [cell.value, cell.ci_low, cell.ci_high].filter((v): v is number => v != null),
  );
  // Zoom in on the data: a 0..1 track would squash differences of a few points.
  let [d0, d1] = metricDomain(metric, values);
  if (metricSpec(metric).format === "percent" || metricSpec(metric).format === "ratio") {
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const pad = Math.max((hi - lo) * 0.15, 0.01);
    d0 = Math.max(0, lo - pad);
    d1 = Math.min(1, hi + pad);
  }
  return (
    <figure className="space-y-1.5">
      <figcaption className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
        {metricLabel(metric)} with 95% CI
      </figcaption>
      {cells.map(({ row, cell }) => (
        <div key={row.name} className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_4.5rem] items-center gap-3 text-xs">
          <span className="truncate text-zinc-600 dark:text-zinc-400" title={row.name}>
            {row.name}
          </span>
          <CIBar
            value={cell.value}
            low={cell.ci_low}
            high={cell.ci_high}
            domain={[d0, d1]}
            label={`${row.name}: ${formatMetric(metric, cell.value)} ${formatCI(metric, cell.ci_low, cell.ci_high)}`}
          />
          <span className="text-right font-medium tabular-nums">{formatMetric(metric, cell.value)}</span>
        </div>
      ))}
      <div className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_4.5rem] gap-3 text-[10px] text-zinc-500 tabular-nums dark:text-zinc-400">
        <span />
        <span className="flex justify-between">
          <span>{formatMetric(metric, d0)}</span>
          <span>{formatMetric(metric, d1)}</span>
        </span>
        <span />
      </div>
    </figure>
  );
}

export function ResultTableView({ table }: { table: ResultTable }) {
  const headingId = `table-${table.id}`;
  return (
    <Card className="overflow-hidden" labelledBy={headingId}>
      <div className="space-y-4 p-4 sm:p-5">
        <SectionHeader id={headingId} title={table.title} description={table.description} />
        <WhiskerChart table={table} />
      </div>
      <div className="overflow-x-auto border-t border-zinc-200 dark:border-zinc-800">
        <table className="w-full text-sm">
          <thead className="bg-zinc-50 text-xs text-zinc-500 dark:bg-zinc-900/60 dark:text-zinc-400">
            <tr>
              <th scope="col" className="px-4 py-2 text-left font-medium sm:pl-5">
                Variant
              </th>
              {table.columns.map((c) => (
                <th key={c} scope="col" className="px-4 py-2 text-right font-medium whitespace-nowrap">
                  {metricSpec(c).short ?? metricLabel(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
            {table.rows.map((row) => {
              const isBaseline = row.name === table.baseline;
              return (
                <tr key={row.name} className={isBaseline ? "bg-zinc-50/60 dark:bg-zinc-900/40" : undefined}>
                  <th scope="row" className="px-4 py-2.5 text-left font-medium sm:pl-5">
                    <a
                      href={href.run(row.run_id)}
                      className="hover:text-accent-700 hover:underline dark:hover:text-accent-300"
                    >
                      {row.name}
                    </a>
                    {isBaseline ? (
                      <Badge className="ml-2" tone="neutral">
                        baseline
                      </Badge>
                    ) : null}
                  </th>
                  {table.columns.map((c) => {
                    const cell = row.metrics[c];
                    const ci = cell ? formatCI(c, cell.ci_low, cell.ci_high) : "";
                    return (
                      <td key={c} className="px-4 py-2.5 text-right align-top whitespace-nowrap tabular-nums">
                        {cell ? (
                          <div className="flex flex-col items-end">
                            <span className="font-medium">{formatMetric(c, cell.value)}</span>
                            {ci ? <span className="text-[11px] text-zinc-500 dark:text-zinc-400">{ci}</span> : null}
                            <DeltaText metric={c} cell={cell} />
                          </div>
                        ) : (
                          <span className="text-zinc-500 dark:text-zinc-400">n/a</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
