import type { MetricValue } from "../../api/types";
import { formatCI, formatMetric, metricDomain, metricSpec } from "../../lib/metrics";
import { Card } from "../ui";

export function CIBar({
  value,
  low,
  high,
  domain,
  label,
}: {
  value: number;
  low?: number | null;
  high?: number | null;
  domain: [number, number];
  label: string;
}) {
  const [d0, d1] = domain;
  const x = (v: number) => `${Math.min(100, Math.max(0, ((v - d0) / (d1 - d0 || 1)) * 100))}%`;
  const hasCI = low != null && high != null;
  return (
    <svg className="h-3 w-full overflow-visible" role="img" aria-label={label}>
      <line x1="0" x2="100%" y1="6" y2="6" className="stroke-zinc-200 dark:stroke-zinc-800" strokeWidth="4" strokeLinecap="round" />
      {hasCI ? (
        <line x1={x(low)} x2={x(high)} y1="6" y2="6" className="stroke-accent-600/35 dark:stroke-accent-400/40" strokeWidth="4" strokeLinecap="round" />
      ) : null}
      <circle cx={x(value)} cy="6" r="4" className="fill-accent-700 stroke-white dark:fill-accent-400 dark:stroke-zinc-900" strokeWidth="1.5" />
    </svg>
  );
}

export function MetricCard({ name, metric }: { name: string; metric: MetricValue }) {
  const spec = metricSpec(name);
  const ci = formatCI(name, metric.ci_low, metric.ci_high);
  const domain = metricDomain(
    name,
    [metric.value, metric.ci_low, metric.ci_high].filter((v): v is number => v != null),
  );
  return (
    <Card as="div" className="flex flex-col gap-2 p-4">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="truncate text-xs font-medium text-zinc-600 dark:text-zinc-400" title={spec.help}>
          {spec.label}
        </h3>
        <span className="text-[11px] text-zinc-400 tabular-nums dark:text-zinc-500">n={metric.n}</span>
      </div>
      <p className="text-2xl font-semibold tracking-tight tabular-nums">{formatMetric(name, metric.value)}</p>
      <p className="h-4 text-xs text-zinc-500 tabular-nums dark:text-zinc-400">
        {ci ? <span title="95% confidence interval">95% CI {ci}</span> : spec.format === "usd" ? "list-price equivalent" : null}
      </p>
      <CIBar
        value={metric.value}
        low={metric.ci_low}
        high={metric.ci_high}
        domain={domain}
        label={`${spec.label} ${formatMetric(name, metric.value)}${ci ? `, 95% CI ${ci}` : ""}`}
      />
    </Card>
  );
}
