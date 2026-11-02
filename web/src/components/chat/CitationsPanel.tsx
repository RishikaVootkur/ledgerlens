import type { Citation } from "../../api/types";
import { parseFilingId } from "../../lib/citations";
import { cx } from "../../lib/style";
import { ChunkViewer } from "./ChunkViewer";

export function CitationList({
  citations,
  activeIndex,
  onSelect,
  companyName,
}: {
  citations: Citation[];
  activeIndex: number | null;
  onSelect: (index: number) => void;
  companyName: (ticker: string) => string | undefined;
}) {
  return (
    <ol className="space-y-1.5">
      {citations.map((c, i) => {
        const index = i + 1;
        const f = parseFilingId(c.filing_id);
        const active = activeIndex === index;
        const pages = c.page ? (c.page_end && c.page_end !== c.page ? `pp. ${c.page}-${c.page_end}` : `p. ${c.page}`) : null;
        return (
          <li key={`${c.chunk_id}-${i}`}>
            <button
              type="button"
              onClick={() => onSelect(index)}
              aria-pressed={active}
              className={cx(
                "flex w-full gap-3 rounded-lg border px-3 py-2 text-left transition-colors",
                active
                  ? "border-accent-300 bg-accent-50/70 dark:border-accent-800 dark:bg-accent-950/40"
                  : "border-zinc-200 hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-800/50",
              )}
            >
              <span
                className={cx(
                  "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded text-[0.7rem] font-semibold tabular-nums",
                  active ? "bg-accent-700 text-white dark:bg-accent-500 dark:text-zinc-950" : "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
                )}
                aria-hidden="true"
              >
                {index}
              </span>
              <span className="min-w-0 text-sm">
                <span className="block font-medium text-zinc-900 dark:text-zinc-100">
                  {f ? (companyName(f.ticker) ?? f.ticker) : c.filing_id}
                </span>
                <span className="block text-xs text-zinc-600 tabular-nums dark:text-zinc-400">
                  {[f?.form, f?.period, pages].filter(Boolean).join(" · ")}
                </span>
                <span className="mt-0.5 block truncate text-xs text-zinc-500 dark:text-zinc-500" title={c.section}>
                  {c.section}
                </span>
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

export function CitationsPanel({
  citations,
  activeIndex,
  onSelect,
  companyName,
  pending,
}: {
  citations: Citation[];
  activeIndex: number | null;
  onSelect: (index: number) => void;
  companyName: (ticker: string) => string | undefined;
  pending: boolean;
}) {
  const active = activeIndex !== null ? citations[activeIndex - 1] : undefined;
  if (citations.length === 0) {
    return (
      <p className="text-sm text-zinc-500 dark:text-zinc-400">
        {pending ? "Sources appear when the answer is complete." : "Sources for the answer appear here."}
      </p>
    );
  }
  return (
    <div className="space-y-4">
      <CitationList citations={citations} activeIndex={activeIndex} onSelect={onSelect} companyName={companyName} />
      {active ? (
        <div className="border-t border-zinc-200 pt-4 dark:border-zinc-800">
          <ChunkViewer citation={active} />
        </div>
      ) : (
        <p className="text-xs text-zinc-500 dark:text-zinc-400">Select a source to read the exact passage.</p>
      )}
    </div>
  );
}
