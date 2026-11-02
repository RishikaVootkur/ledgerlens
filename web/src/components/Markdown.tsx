import { useMemo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation } from "../api/types";
import { citationLabel, citeIndexFromHref, linkifyCitations } from "../lib/citations";
import { rehypeHighlightQuote } from "../lib/rehypeHighlight";
import { cx } from "../lib/style";

const tableComponents: Components = {
  table: ({ children }) => (
    <div className="table-wrap">
      <table>{children}</table>
    </div>
  ),
};

export function CitationChip({
  index,
  citation,
  active,
  onSelect,
}: {
  index: number;
  citation?: Citation;
  active?: boolean;
  onSelect?: (index: number) => void;
}) {
  const label = citation ? `Source ${index}: ${citationLabel(citation)}` : `Source ${index}`;
  return (
    <button
      type="button"
      className={cx(
        "cite-chip mx-0.5 inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-px items-center justify-center rounded px-1 align-middle text-[0.7rem] font-semibold tabular-nums ring-1 ring-inset transition-colors",
        active
          ? "bg-accent-700 text-white ring-accent-700 dark:bg-accent-500 dark:text-zinc-950 dark:ring-accent-500"
          : "bg-accent-50 text-accent-800 ring-accent-200 hover:bg-accent-100 dark:bg-accent-950 dark:text-accent-200 dark:ring-accent-800 dark:hover:bg-accent-900",
      )}
      aria-label={label}
      aria-pressed={active}
      title={citation ? citationLabel(citation) : undefined}
      disabled={!onSelect}
      onClick={() => onSelect?.(index)}
    >
      {index}
    </button>
  );
}

/** Answer text with `[n]` markers rendered as clickable source chips. */
export function AnswerMarkdown({
  text,
  citations,
  activeIndex,
  onSelectCitation,
  streaming = false,
}: {
  text: string;
  /** Undefined while streaming, before the final answer lists its citations. */
  citations?: Citation[];
  activeIndex?: number | null;
  onSelectCitation?: (index: number) => void;
  streaming?: boolean;
}) {
  const source = useMemo(
    () => linkifyCitations(text, citations ? citations.length : Number.POSITIVE_INFINITY),
    [text, citations],
  );
  const components = useMemo<Components>(
    () => ({
      ...tableComponents,
      a: ({ href, children }) => {
        const index = citeIndexFromHref(href);
        if (index !== null) {
          return (
            <CitationChip
              index={index}
              citation={citations?.[index - 1]}
              active={activeIndex === index}
              onSelect={citations ? onSelectCitation : undefined}
            />
          );
        }
        return (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        );
      },
    }),
    [citations, activeIndex, onSelectCitation],
  );
  return (
    <div className={cx("md", streaming && "streaming-caret")}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {source}
      </ReactMarkdown>
    </div>
  );
}

/** A filing passage with GitHub tables rendered and the cited quote highlighted. */
export function ChunkMarkdown({ text, quote }: { text: string; quote: string }) {
  return (
    <div className="md md-compact">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlightQuote, { quote }]]}
        components={tableComponents}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
