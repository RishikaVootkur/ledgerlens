import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import retrievalRun from "../../mock/runs/2026-10-27T0910-bm25.json";
import fullRun from "../../mock/runs/2026-10-28T1402-full-gemini-flash.json";
import { RunPage } from "./RunPage";

function serve(body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } })),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("RunPage", () => {
  it("shows retrieval columns and no outcome filter for retrieval-only runs", async () => {
    serve(retrievalRun);
    render(<RunPage runId={retrievalRun.run_id} qid={null} />);
    expect(await screen.findByText(/retrieval only/)).toBeInTheDocument();
    expect(screen.queryByLabelText("Outcome")).toBeNull();
    expect(screen.getByRole("columnheader", { name: "Recall@5" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Outcome" })).toBeNull();
  });

  it("shows outcomes for runs with answers", async () => {
    serve(fullRun);
    render(<RunPage runId={fullRun.run_id} qid={null} />);
    expect(await screen.findByLabelText("Outcome")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Outcome" })).toBeInTheDocument();
    expect(screen.getAllByText("Unscored").length).toBeGreaterThan(0);
  });
});
