import type { JudgeCalibration, MetricValue } from "../../api/types";
import { formatCI, formatMetric } from "../../lib/metrics";
import { Card, SectionHeader } from "../ui";

function labelName(label: string): string {
  return label.charAt(0).toUpperCase() + label.slice(1).toLowerCase().replace(/_/g, " ");
}

function calibrationNote(split: string): string {
  return split === "test"
    ? "Calibrated against FinanceBench human labels (held-out split)."
    : `Calibrated against FinanceBench human labels (split: ${split}).`;
}

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
  const { confusion, labels } = calibration;
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
      <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">
        <Stat name="judge_correct" label={`${labels.length}-way agreement`} metric={calibration.accuracy} />
        <Stat name="kappa" label={`${labels.length}-way kappa`} metric={calibration.kappa} />
        <Stat name="error_rate" label="Judge error rate" metric={calibration.error_rate} />
        <Stat name="judge_correct" label="Binary agreement" metric={calibration.binary_accuracy} />
        <Stat name="kappa" label="Binary kappa" metric={calibration.binary_kappa} />
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
                <th key={l} scope="col" className="px-1 pb-1 text-xs font-medium text-zinc-500 dark:text-zinc-400">
                  <span className="sr-only">Judge: </span>
                  {labelName(l)}
                </th>
              ))}
              <th scope="col" className="px-2 pb-1 text-xs font-medium text-zinc-500 dark:text-zinc-400">
                Recall
              </th>
            </tr>
          </thead>
          <tbody>
            {confusion.map((row, i) => {
              const label = labels[i] ?? `Class ${i + 1}`;
              const recall = calibration.per_label_recall[label];
              return (
                <tr key={label}>
                  <th
                    scope="row"
                    className="pr-2 text-right text-xs font-medium whitespace-nowrap text-zinc-500 dark:text-zinc-400"
                  >
                    <span className="sr-only">Human: </span>
                    {labelName(label)}
                  </th>
                  {row.map((count, j) => {
                    const agree = i === j;
                    const strength = count / max;
                    return (
                      <td key={j} className="p-0.5">
                        <div
                          className={
                            agree
                              ? "flex h-10 min-w-14 items-center justify-center rounded-md font-semibold text-accent-950 dark:text-accent-50"
                              : "flex h-10 min-w-14 items-center justify-center rounded-md font-semibold text-rose-900 dark:text-rose-100"
                          }
                          style={{
                            backgroundColor: agree
                              ? `color-mix(in oklab, var(--color-accent-500) ${Math.round(12 + strength * 38)}%, transparent)`
                              : `color-mix(in oklab, var(--color-rose-500) ${Math.round(6 + strength * 40)}%, transparent)`,
                          }}
                        >
                          {count}
                        </div>
                      </td>
                    );
                  })}
                  <td className="px-2 text-right text-xs text-zinc-600 dark:text-zinc-400">
                    {recall != null ? formatMetric("judge_correct", recall) : "n/a"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {calibration.clean_n && calibration.clean_accuracy && calibration.clean_kappa ? (
        <p className="mt-4 text-sm text-zinc-700 tabular-nums dark:text-zinc-300">
          Untouched by prompt tuning (n={calibration.clean_n}): agreement{" "}
          <span className="font-semibold">{formatMetric("judge_correct", calibration.clean_accuracy.value)}</span>{" "}
          {formatCI("judge_correct", calibration.clean_accuracy.ci_low, calibration.clean_accuracy.ci_high)}, kappa{" "}
          <span className="font-semibold">{formatMetric("kappa", calibration.clean_kappa.value)}</span>{" "}
          {formatCI("kappa", calibration.clean_kappa.ci_low, calibration.clean_kappa.ci_high)}.
        </p>
      ) : null}
      <p className="mt-4 text-xs text-zinc-500 dark:text-zinc-400">
        {calibrationNote(calibration.split)} Judge prompt{" "}
        <span className="font-mono">{calibration.prompt_version}</span>.
      </p>
    </Card>
  );
}
