import { useState } from "react";
import type { FailureExample } from "../../api/types";
import { failureLabel } from "../../lib/failures";
import { href } from "../../lib/router";
import { Badge, Card, SectionHeader } from "../ui";
import { cx } from "../../lib/style";

export function FailureTaxonomy({
  counts,
  failures,
}: {
  counts: Record<string, number>;
  failures: FailureExample[];
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((s, [, n]) => s + n, 0);
  const max = Math.max(1, ...entries.map(([, n]) => n));
  const shown = selected ? failures.filter((f) => f.failure_category === selected) : failures;

  return (
    <Card className="p-4 sm:p-5" labelledBy="failures-heading">
      <SectionHeader
        id="failures-heading"
        title="Failure taxonomy"
        description={`${total} labeled failures from the headline run. Select a category to filter the examples.`}
      />
      <ul className="mt-4 space-y-1.5" aria-label="Failure counts by category">
        {entries.map(([category, n]) => {
          const active = selected === category;
          return (
            <li key={category}>
              <button
                type="button"
                aria-pressed={active}
                onClick={() => setSelected(active ? null : category)}
                className={cx(
                  "grid w-full grid-cols-[minmax(0,10rem)_minmax(0,1fr)_3.5rem] items-center gap-3 rounded-md px-2 py-1 text-left text-sm",
                  active ? "bg-accent-50 dark:bg-accent-950/50" : "hover:bg-zinc-50 dark:hover:bg-zinc-800/50",
                )}
              >
                <span className="truncate text-zinc-700 dark:text-zinc-300">{failureLabel(category)}</span>
                <span className="h-2.5 rounded-sm bg-zinc-100 dark:bg-zinc-800" aria-hidden="true">
                  <span
                    className={cx(
                      "block h-full rounded-sm",
                      active ? "bg-accent-700 dark:bg-accent-400" : "bg-accent-600/70 dark:bg-accent-500/70",
                    )}
                    style={{ width: `${(n / max) * 100}%` }}
                  />
                </span>
                <span className="text-right text-zinc-600 tabular-nums dark:text-zinc-400">
                  {n}
                  <span className="ml-1 text-[11px] text-zinc-500 dark:text-zinc-400">
                    {total ? `${Math.round((n / total) * 100)}%` : ""}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>

      <div className="mt-5">
        <h3 className="text-xs font-medium tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
          Examples{selected ? `: ${failureLabel(selected)}` : ""} ({shown.length})
        </h3>
        <div className="mt-2 divide-y divide-zinc-200 rounded-lg border border-zinc-200 dark:divide-zinc-800 dark:border-zinc-800">
          {shown.length === 0 ? (
            <p className="px-3 py-3 text-sm text-zinc-500 dark:text-zinc-400">No examples for this category.</p>
          ) : null}
          {shown.map((f) => (
            <details key={`${f.run_id}-${f.qid}`} className="group px-3 py-2.5 text-sm">
              <summary className="flex cursor-pointer list-none items-start gap-2 rounded [&::-webkit-details-marker]:hidden">
                <svg
                  viewBox="0 0 12 12"
                  className="mt-1 size-3 shrink-0 text-zinc-500 dark:text-zinc-400 transition-transform group-open:rotate-90"
                  aria-hidden="true"
                >
                  <path d="M4 2.5L7.5 6 4 9.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                <span className="min-w-0 flex-1">{f.question}</span>
                <Badge className="hidden sm:inline-flex">{failureLabel(f.failure_category)}</Badge>
              </summary>
              <dl className="mt-2 grid gap-x-4 gap-y-1.5 pl-5 text-sm sm:grid-cols-[6rem_minmax(0,1fr)]">
                <dt className="text-zinc-500 dark:text-zinc-400">Gold</dt>
                <dd className="tabular-nums">{f.gold ?? "Unanswerable"}</dd>
                <dt className="text-zinc-500 dark:text-zinc-400">Answer</dt>
                <dd className="tabular-nums">{f.answer ?? "No answer"}</dd>
                {f.explanation ? (
                  <>
                    <dt className="text-zinc-500 dark:text-zinc-400">Why</dt>
                    <dd className="text-zinc-700 dark:text-zinc-300">{f.explanation}</dd>
                  </>
                ) : null}
                <dt className="text-zinc-500 dark:text-zinc-400">Source</dt>
                <dd>
                  <a
                    href={href.run(f.run_id, f.qid)}
                    className="font-mono text-xs text-accent-700 hover:underline dark:text-accent-300"
                  >
                    {f.qid} · {f.dataset}
                  </a>
                </dd>
              </dl>
            </details>
          ))}
        </div>
      </div>
    </Card>
  );
}
