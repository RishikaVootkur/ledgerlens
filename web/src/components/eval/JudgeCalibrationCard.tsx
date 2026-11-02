import type { JudgeCalibration, MetricValue } from "../../api/types";
import { formatCI, formatMetric } from "../../lib/metrics";
import { Card, SectionHeader } from "../ui";

const BINARY_LABELS = ["Correct", "Incorrect"];

function Stat({ name, label, metric }: { name: string; label: string; metric: MetricValue }) {
  const ci = formatCI(name, metric.ci_low, metric.ci_high);
  return (
    <div>
      <dt className="text-xs text-zinc-500 dark:text-zinc-400">{label}</dt>
      <dd className="mt-0.5 text-xl font-semibold tracking-tight tabular-nums">{formatMetric(name, metric.value)}</dd>
      {ci ? <dd className="text-xs text-zinc-500 tabular-nums dark:text-zinc-400">95% CI {ci}</dd> : null}
    </div>
  );
}

export function JudgeCalibrationCard({ calibration }: { calibration: JudgeCalibration }) {
  const { confusion } = calibration;
  const size = confusion.length;
  const labels = size === 2 ? BINARY_LABELS : confusion.map((_, i) => `Class ${i + 1}`);
  const max = Math.max(1, ...confusion.flat());
  return (
    <Card className="p-4 sm:p-5" labelledBy="judge-heading">
      <SectionHeader
        id="judge-heading"
        title="Judge calibration"
        description={
          <>
            Agreement of <span className="font-mono text-xs">{calibration.judge_model}</span> with{" "}
            {calibration.n} human labels.
          </>
        }
      />
      <dl className="mt-4 grid grid-cols-2 gap-4">
        <Stat name="judge_accuracy" label="Agreement" metric={calibration.accuracy} />
        <Stat name="kappa" label="Cohen's kappa" metric={calibration.kappa} />
      </dl>
      <div className="relative mt-5 overflow-x-auto">
        <table className="text-sm tabular-nums">
          <caption className="mb-2 text-left text-xs text-zinc-500 dark:text-zinc-400">
            Confusion matrix: rows are the human label, columns the judge label
          </caption>
          <thead>
            <tr>
              <td />
              {labels.map((l) => (
                <th key={l} scope="col" className="px-2 pb-1 text-xs font-medium text-zinc-500 dark:text-zinc-400">
                  Judge: {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {confusion.map((row, i) => (
              <tr key={labels[i]}>
                <th scope="row" className="pr-3 text-right text-xs font-medium whitespace-nowrap text-zinc-500 dark:text-zinc-400">
                  Human: {labels[i]}
                </th>
                {row.map((count, j) => {
                  const agree = i === j;
                  const strength = count / max;
                  return (
                    <td key={j} className="p-1">
                      <div
                        className={
                          agree
                            ? "flex h-12 min-w-20 items-center justify-center rounded-md font-semibold text-accent-950 dark:text-accent-50"
                            : "flex h-12 min-w-20 items-center justify-center rounded-md font-semibold text-rose-900 dark:text-rose-100"
                        }
                        style={{
                          backgroundColor: agree
                            ? `color-mix(in oklab, var(--color-accent-500) ${Math.round(12 + strength * 38)}%, transparent)`
                            : `color-mix(in oklab, var(--color-rose-500) ${Math.round(8 + strength * 40)}%, transparent)`,
                        }}
                      >
                        {count}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
