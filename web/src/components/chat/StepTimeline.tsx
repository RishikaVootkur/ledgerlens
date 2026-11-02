import type { StepState } from "../../lib/chatState";
import { Spinner } from "../ui";
import { cx } from "../../lib/style";

const STEP_LABELS: Record<string, string> = {
  plan: "Plan",
  search_filings: "Search filings",
  get_company_facts: "XBRL company facts",
  verify: "Verify numbers",
  answer: "Write answer",
  self_check: "Self-check",
};

function stepLabel(name: string): string {
  return STEP_LABELS[name] ?? name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}

export function StepTimeline({ steps, live }: { steps: StepState[]; live: boolean }) {
  if (steps.length === 0) return null;
  return (
    <ol className="space-y-0" aria-label="Agent steps" aria-live={live ? "polite" : undefined}>
      {steps.map((step, i) => {
        const last = i === steps.length - 1;
        const active = step.status === "active" && live;
        return (
          <li key={`${step.name}-${i}`} className="relative flex gap-3 pb-3 last:pb-0">
            {!last ? (
              <span
                className="absolute top-5 left-[9px] h-[calc(100%-1rem)] w-px bg-zinc-200 dark:bg-zinc-800"
                aria-hidden="true"
              />
            ) : null}
            <span
              className={cx(
                "relative z-10 mt-0.5 flex size-[19px] shrink-0 items-center justify-center rounded-full ring-1",
                active
                  ? "bg-white text-accent-700 ring-accent-300 dark:bg-zinc-900 dark:text-accent-300 dark:ring-accent-700"
                  : "bg-accent-50 text-accent-700 ring-accent-200 dark:bg-accent-950 dark:text-accent-300 dark:ring-accent-800",
              )}
            >
              {active ? (
                <Spinner className="size-3" label="In progress" />
              ) : (
                <svg viewBox="0 0 12 12" className="size-2.5" aria-label="Done" role="img">
                  <path d="M2.5 6.5l2.2 2L9.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </span>
            <div className="min-w-0 text-sm">
              <span className={cx("font-medium", active ? "text-zinc-900 dark:text-zinc-100" : "text-zinc-700 dark:text-zinc-300")}>
                {stepLabel(step.name)}
              </span>
              {step.detail ? (
                <p className="mt-0.5 text-xs leading-snug [overflow-wrap:anywhere] text-zinc-500 dark:text-zinc-400">{step.detail}</p>
              ) : null}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
