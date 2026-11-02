import { useState, type ReactNode } from "react";
import type { QuestionResult } from "../../api/types";
import { citationLabel } from "../../lib/citations";
import { failureLabel } from "../../lib/failures";
import { formatMetric, metricLabel, orderMetrics } from "../../lib/metrics";
import { questionOutcome } from "../../lib/runs";
import { UsageBar } from "../chat/UsageBar";
import { AnswerMarkdown } from "../Markdown";
import { Badge } from "../ui";
import { cx } from "../../lib/style";
import { OutcomeBadge } from "./OutcomeBadge";

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <section>
      <h3 className="mb-1 text-xs font-medium tracking-wide text-zinc-500 uppercase dark:text-zinc-400">{label}</h3>
      {children}
    </section>
  );
}

export function QuestionDetail({ result }: { result: QuestionResult }) {
  const [active, setActive] = useState<number | null>(null);
  const prediction = result.prediction;
  const citations = prediction?.citations ?? [];
  const scoreNames = orderMetrics(Object.keys(result.scores));

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-1.5">
        <OutcomeBadge outcome={questionOutcome(result)} />
        <Badge>{result.dataset}</Badge>
        {result.question_type ? <Badge>{result.question_type.replace(/_/g, " ")}</Badge> : null}
        {result.ticker ? <Badge tone="accent">{result.ticker}</Badge> : null}
        {result.fiscal_year ? <Badge>FY{result.fiscal_year}</Badge> : null}
        {!result.answerable ? <Badge tone="warn">unanswerable</Badge> : null}
        {result.failure_category ? <Badge tone="bad">{failureLabel(result.failure_category)}</Badge> : null}
      </div>

      <Field label="Question">
        <p className="text-[0.9375rem] leading-relaxed">{result.question}</p>
      </Field>

      <Field label="Gold">
        <p className="text-sm tabular-nums">{result.gold ?? <span className="text-zinc-500">None (should refuse)</span>}</p>
      </Field>

      <Field label="Answer">
        {result.error ? (
          <pre className="overflow-x-auto rounded-md bg-rose-50 p-2 text-xs whitespace-pre-wrap text-rose-900 dark:bg-rose-950/50 dark:text-rose-200">
            {result.error}
          </pre>
        ) : prediction ? (
          <div className="space-y-2">
            {prediction.refused ? <Badge tone="warn">Refused: insufficient support</Badge> : null}
            <AnswerMarkdown
              text={prediction.answer}
              citations={citations}
              activeIndex={active}
              onSelectCitation={(i) => setActive((cur) => (cur === i ? null : i))}
            />
          </div>
        ) : (
          <p className="text-sm text-zinc-500">No prediction.</p>
        )}
      </Field>

      {result.failure_note ? (
        <Field label="Failure note">
          <p className="text-sm text-zinc-700 dark:text-zinc-300">{result.failure_note}</p>
        </Field>
      ) : null}

      {citations.length > 0 ? (
        <Field label={`Citations (${citations.length})`}>
          <ol className="space-y-2">
            {citations.map((c, i) => (
              <li
                key={`${c.chunk_id}-${i}`}
                className={cx(
                  "rounded-lg border px-3 py-2 text-sm",
                  active === i + 1
                    ? "border-accent-300 bg-accent-50/60 dark:border-accent-800 dark:bg-accent-950/40"
                    : "border-zinc-200 dark:border-zinc-800",
                )}
              >
                <p className="font-medium">
                  <span className="mr-1.5 text-zinc-400 tabular-nums">{i + 1}.</span>
                  {citationLabel(c)}
                </p>
                <p className="truncate text-xs text-zinc-500 dark:text-zinc-400" title={c.section}>
                  {c.section}
                </p>
                {c.quote ? (
                  <p className="mt-1 border-l-2 border-amber-400 pl-2 text-xs text-zinc-700 italic dark:text-zinc-300">
                    “{c.quote}”
                  </p>
                ) : null}
              </li>
            ))}
          </ol>
        </Field>
      ) : null}

      <Field label="Scores">
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-3">
          {scoreNames.map((name) => (
            <div key={name} className="flex justify-between gap-2 border-b border-zinc-100 py-1 dark:border-zinc-800">
              <dt className="truncate text-zinc-500 dark:text-zinc-400">{metricLabel(name)}</dt>
              <dd className="font-medium tabular-nums">{formatMetric(name, result.scores[name])}</dd>
            </div>
          ))}
        </dl>
      </Field>

      {prediction ? (
        <Field label="Usage">
          <UsageBar usage={prediction.usage} />
          {result.wall_ms != null ? (
            <p className="mt-1 text-xs text-zinc-500 tabular-nums dark:text-zinc-400">
              Wall time {formatMetric("latency_p50_ms", result.wall_ms)}
            </p>
          ) : null}
        </Field>
      ) : null}

      <p className="font-mono text-[11px] text-zinc-400 dark:text-zinc-500">{result.qid}</p>
    </div>
  );
}
