import { describe, expect, it } from "vitest";
import {
  citationLabel,
  citeIndexFromHref,
  findQuote,
  linkifyCitations,
  parseCitationMarkers,
  parseFilingId,
} from "./citations";

describe("parseCitationMarkers", () => {
  it("splits text and single markers", () => {
    expect(parseCitationMarkers("Revenue was $391B [1]. Services grew [2].", 2)).toEqual([
      { type: "text", value: "Revenue was $391B " },
      { type: "cite", indices: [1], raw: "[1]" },
      { type: "text", value: ". Services grew " },
      { type: "cite", indices: [2], raw: "[2]" },
      { type: "text", value: "." },
    ]);
  });

  it("supports grouped and adjacent markers", () => {
    const segs = parseCitationMarkers("A [1, 3] B [2][3]", 3).filter((s) => s.type === "cite");
    expect(segs.map((s) => (s.type === "cite" ? s.indices : []))).toEqual([[1, 3], [2], [3]]);
  });

  it("leaves out-of-range markers and Markdown links as text", () => {
    expect(parseCitationMarkers("See [4] and [1](http://x) and [2]: note", 3)).toEqual([
      { type: "text", value: "See [4] and [1](http://x) and [2]: note" },
    ]);
    expect(parseCitationMarkers("Zero [0]", 3)).toEqual([{ type: "text", value: "Zero [0]" }]);
  });

  it("accepts any index while streaming (count = Infinity)", () => {
    expect(parseCitationMarkers("x [7]", Number.POSITIVE_INFINITY)[1]).toEqual({
      type: "cite",
      indices: [7],
      raw: "[7]",
    });
  });
});

describe("linkifyCitations", () => {
  it("rewrites markers into chip links that round-trip", () => {
    expect(linkifyCitations("Up 2% [1, 2].", 2)).toBe("Up 2% [1](#cite-1)[2](#cite-2).");
    expect(citeIndexFromHref("#cite-2")).toBe(2);
    expect(citeIndexFromHref("https://sec.gov")).toBeNull();
    expect(citeIndexFromHref(undefined)).toBeNull();
  });
});

describe("findQuote", () => {
  it("matches case-insensitively across whitespace differences", () => {
    const text = "Total net sales\nincreased 2% or $7.8 billion during 2024.";
    expect(findQuote(text, "total NET sales increased 2%")).toEqual({ start: 0, end: 28 });
  });

  it("escapes regex characters and falls back to null", () => {
    expect(findQuote("Margin (GAAP) was 46.2%", "(GAAP) was 46.2%")).not.toBeNull();
    expect(findQuote("abc", "xyz")).toBeNull();
    expect(findQuote("abc", "")).toBeNull();
  });
});

describe("filing labels", () => {
  it("parses filing ids and formats citation labels", () => {
    expect(parseFilingId("AAPL-10K-2024FY")).toEqual({ ticker: "AAPL", form: "10-K", period: "FY2024" });
    expect(parseFilingId("JPM-10Q-2025Q2")).toEqual({ ticker: "JPM", form: "10-Q", period: "Q2 2025" });
    expect(parseFilingId("weird")).toBeNull();
    expect(citationLabel({ filing_id: "AAPL-10K-2024FY", page: 22, page_end: 23 })).toBe(
      "AAPL 10-K FY2024, pp. 22-23",
    );
    expect(citationLabel({ filing_id: "KO-10K-2023FY", page: null })).toBe("KO 10-K FY2023");
  });
});
