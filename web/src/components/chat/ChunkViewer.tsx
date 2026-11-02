import { useEffect, useRef } from "react";
import { getChunk } from "../../api/client";
import type { Citation } from "../../api/types";
import { findQuote } from "../../lib/citations";
import { useAsync } from "../../lib/useAsync";
import { ChunkMarkdown } from "../Markdown";
import { Badge, ErrorNote, LoadingBlock } from "../ui";

/** Fetches the cited chunk and shows the exact passage with the quote highlighted. */
export function ChunkViewer({ citation }: { citation: Citation }) {
  const state = useAsync(() => getChunk(citation.chunk_id), citation.chunk_id);
  const bodyRef = useRef<HTMLDivElement>(null);
  const ready = state.status === "ready";

  useEffect(() => {
    if (!ready) return;
    // Scroll only the passage box; scrollIntoView would also move the page.
    const box = bodyRef.current;
    const mark = box?.querySelector<HTMLElement>("mark[data-quote]");
    if (!box || !mark) return;
    const offset = mark.getBoundingClientRect().top - box.getBoundingClientRect().top;
    box.scrollTo?.({ top: box.scrollTop + offset - box.clientHeight / 2, behavior: "smooth" });
  }, [ready, citation.quote]);

  if (state.status === "loading") return <LoadingBlock label="Loading passage" />;
  if (state.status === "error") return <ErrorNote title="Could not load this passage" error={state.error} />;

  const chunk = state.data;
  const located = citation.quote ? findQuote(chunk.text, citation.quote) !== null : false;
  const pages =
    chunk.page != null
      ? chunk.page_end != null && chunk.page_end !== chunk.page
        ? `pp. ${chunk.page}-${chunk.page_end}`
        : `p. ${chunk.page}`
      : null;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone="accent">{chunk.ticker}</Badge>
        <Badge>{chunk.form}</Badge>
        <Badge>
          {chunk.fiscal_period === "FY" ? `FY${chunk.fiscal_year}` : `${chunk.fiscal_period} ${chunk.fiscal_year}`}
        </Badge>
        {pages ? <Badge>{pages}</Badge> : null}
        {chunk.has_table ? <Badge>table</Badge> : null}
      </div>
      <p className="text-xs leading-snug text-zinc-500 dark:text-zinc-400">
        <span className="font-medium text-zinc-700 dark:text-zinc-300">{chunk.company}</span> · {chunk.section}
      </p>
      {citation.quote ? (
        <blockquote className="border-l-2 border-amber-400 pl-3 text-sm text-zinc-700 italic dark:text-zinc-300">
          “{citation.quote}”
          {!located ? (
            <span className="mt-1 block text-xs text-zinc-500 not-italic dark:text-zinc-400">
              Exact quote not found in the passage text, so nothing is highlighted.
            </span>
          ) : null}
        </blockquote>
      ) : null}
      <div
        ref={bodyRef}
        className="max-h-[55vh] overflow-y-auto rounded-lg border border-zinc-200 bg-zinc-50/60 p-3 dark:border-zinc-800 dark:bg-zinc-950/40"
      >
        <ChunkMarkdown text={chunk.text} quote={citation.quote} />
      </div>
      <p className="font-mono text-[11px] break-all text-zinc-500 dark:text-zinc-400">{chunk.chunk_id}</p>
    </div>
  );
}
