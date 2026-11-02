// Inline citation markers: the agent writes `[1]`, `[2]` or `[1, 3]` after a claim,
// where the number is the 1-based position in AgentAnswer.citations.

export type AnswerSegment =
  | { type: "text"; value: string }
  | { type: "cite"; indices: number[]; raw: string };

// `[1]`, `[1,2]`, `[1, 2]`; not a Markdown link `[1](...)` or reference definition `[1]: ...`.
const MARKER = /\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\](?![(:])/g;

/**
 * Split answer text into plain text and citation markers. Markers that point
 * outside 1..count stay as plain text, so a bad index never becomes a dead chip.
 * Pass count = Infinity while streaming, before the citations are known.
 */
export function parseCitationMarkers(text: string, count: number): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    const indices = match[1]!.split(",").map((s) => Number.parseInt(s.trim(), 10));
    if (indices.some((i) => i < 1 || i > count)) continue;
    if (match.index > last) segments.push({ type: "text", value: text.slice(last, match.index) });
    segments.push({ type: "cite", indices, raw: match[0] });
    last = match.index + match[0].length;
  }
  if (last < text.length) segments.push({ type: "text", value: text.slice(last) });
  return segments;
}

export const CITE_HREF_PREFIX = "#cite-";

/**
 * Rewrite markers as Markdown links (`[1](#cite-1)`) so react-markdown keeps
 * them inline in the right place; the link renderer turns them into chips.
 */
export function linkifyCitations(text: string, count: number): string {
  return parseCitationMarkers(text, count)
    .map((seg) =>
      seg.type === "text"
        ? seg.value
        : seg.indices.map((i) => `[${i}](${CITE_HREF_PREFIX}${i})`).join(""),
    )
    .join("");
}

/** Index from a chip href, or null if the href is not a citation link. */
export function citeIndexFromHref(href: string | undefined): number | null {
  if (!href?.startsWith(CITE_HREF_PREFIX)) return null;
  const n = Number.parseInt(href.slice(CITE_HREF_PREFIX.length), 10);
  return Number.isFinite(n) ? n : null;
}

/** "AAPL-10K-2024FY" -> { ticker: "AAPL", form: "10-K", period: "FY2024" }. */
export function parseFilingId(
  filingId: string,
): { ticker: string; form: string; period: string } | null {
  const m = /^([A-Z0-9.]+)-([A-Z0-9]+)-(\d{4})(FY|Q[1-4])$/.exec(filingId);
  if (!m) return null;
  const form = m[2]!.replace(/^10(K|Q)$/, "10-$1");
  return { ticker: m[1]!, form, period: m[4] === "FY" ? `FY${m[3]}` : `${m[4]} ${m[3]}` };
}

/** Short human label for a citation, e.g. "AAPL 10-K FY2024, p. 22". */
export function citationLabel(c: {
  filing_id: string;
  page?: number | null;
  page_end?: number | null;
}): string {
  const f = parseFilingId(c.filing_id);
  const base = f ? `${f.ticker} ${f.form} ${f.period}` : c.filing_id;
  if (!c.page) return base;
  const pages =
    c.page_end && c.page_end !== c.page ? `pp. ${c.page}-${c.page_end}` : `p. ${c.page}`;
  return `${base}, ${pages}`;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Case-insensitive, whitespace-tolerant pattern for a quote, or null when the
 * quote is too short to highlight meaningfully.
 */
export function quotePattern(quote: string): RegExp | null {
  const trimmed = quote.trim();
  if (trimmed.length < 3) return null;
  const body = trimmed.split(/\s+/).map(escapeRegExp).join("\\s+");
  return new RegExp(body, "i");
}

/** Whether the quote appears in the chunk text (case-insensitive substring match). */
export function findQuote(text: string, quote: string): { start: number; end: number } | null {
  const pattern = quotePattern(quote);
  if (!pattern) return null;
  const match = pattern.exec(text);
  return match ? { start: match.index, end: match.index + match[0].length } : null;
}
