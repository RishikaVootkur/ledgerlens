import type { ReactNode } from "react";
import { cx } from "../lib/style";

export function Card({
  children,
  className,
  as: Tag = "section",
  labelledBy,
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "article";
  labelledBy?: string;
}) {
  return (
    <Tag
      aria-labelledby={labelledBy}
      className={cx(
        "rounded-xl border border-zinc-200 bg-white shadow-xs dark:border-zinc-800 dark:bg-zinc-900",
        className,
      )}
    >
      {children}
    </Tag>
  );
}

export function SectionHeader({
  id,
  title,
  description,
  aside,
}: {
  id?: string;
  title: string;
  description?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-x-4 gap-y-2">
      <div className="min-w-0">
        <h2 id={id} className="text-base font-semibold tracking-tight">
          {title}
        </h2>
        {description ? (
          <p className="mt-0.5 max-w-3xl text-sm text-zinc-600 dark:text-zinc-400">{description}</p>
        ) : null}
      </div>
      {aside}
    </div>
  );
}

type Tone = "neutral" | "accent" | "good" | "bad" | "warn";

const toneClasses: Record<Tone, string> = {
  neutral: "bg-zinc-100 text-zinc-700 ring-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:ring-zinc-700",
  accent: "bg-accent-50 text-accent-800 ring-accent-200 dark:bg-accent-950 dark:text-accent-200 dark:ring-accent-800",
  good: "bg-emerald-50 text-emerald-800 ring-emerald-200 dark:bg-emerald-950 dark:text-emerald-200 dark:ring-emerald-800",
  bad: "bg-rose-50 text-rose-800 ring-rose-200 dark:bg-rose-950 dark:text-rose-200 dark:ring-rose-800",
  warn: "bg-amber-50 text-amber-900 ring-amber-200 dark:bg-amber-950 dark:text-amber-200 dark:ring-amber-800",
};

export function Badge({ children, tone = "neutral", className }: { children: ReactNode; tone?: Tone; className?: string }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium whitespace-nowrap ring-1 ring-inset",
        toneClasses[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Spinner({ className, label = "Loading" }: { className?: string; label?: string }) {
  return (
    <svg
      className={cx("size-4 animate-spin motion-reduce:animate-none", className)}
      viewBox="0 0 24 24"
      fill="none"
      role="img"
      aria-label={label}
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export function LoadingBlock({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-10 text-sm text-zinc-500 dark:text-zinc-400" role="status">
      <Spinner label={label} />
      <span>{label}…</span>
    </div>
  );
}

export function ErrorNote({ title, error }: { title: string; error: Error | string }) {
  return (
    <div
      role="alert"
      className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-200"
    >
      <p className="font-medium">{title}</p>
      <p className="mt-1 break-words text-rose-800/90 dark:text-rose-200/80">
        {typeof error === "string" ? error : error.message}
      </p>
    </div>
  );
}
