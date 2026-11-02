import type { Usage } from "../../api/types";
import { formatMetric, formatUSD } from "../../lib/metrics";

const nf = new Intl.NumberFormat("en-US");

export function UsageBar({ usage }: { usage: Usage }) {
  const total = usage.prompt_tokens + usage.completion_tokens;
  const items: Array<{ label: string; value: string; title?: string }> = [
    {
      label: "Tokens",
      value: nf.format(total),
      title: `${nf.format(usage.prompt_tokens)} prompt + ${nf.format(usage.completion_tokens)} completion`,
    },
    { label: "LLM calls", value: nf.format(usage.llm_calls) },
    { label: "Latency", value: formatMetric("latency_p50_ms", usage.latency_ms), title: "Sum of model call latencies" },
    { label: "Cost", value: formatUSD(usage.cost_usd), title: "List-price equivalent, even when the provider was free" },
  ];
  return (
    <dl className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-zinc-500 dark:text-zinc-400">
      {items.map((item) => (
        <div key={item.label} className="flex items-baseline gap-1.5" title={item.title}>
          <dt>{item.label}</dt>
          <dd className="font-medium text-zinc-800 tabular-nums dark:text-zinc-200">
            {item.value}
            {item.label === "Cost" ? (
              <span className="ml-1 font-normal text-zinc-500 dark:text-zinc-400">list-price equivalent</span>
            ) : null}
          </dd>
        </div>
      ))}
    </dl>
  );
}
