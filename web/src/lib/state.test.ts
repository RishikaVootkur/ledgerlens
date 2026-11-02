import { describe, expect, it } from "vitest";
import type { QuestionResult } from "../api/types";
import { STREAM_ENDED_EARLY, chatReducer, initialChatState } from "./chatState";
import { parseHash } from "./router";
import { filterResults, questionOutcome } from "./runs";

describe("chatReducer", () => {
  it("tracks steps, tokens and the final answer", () => {
    let s = chatReducer(initialChatState, { type: "start", question: "q" });
    s = chatReducer(s, { type: "event", event: { type: "step", data: { name: "plan", detail: "" } } });
    s = chatReducer(s, { type: "event", event: { type: "step", data: { name: "search_filings", detail: "" } } });
    expect(s.steps.map((x) => x.status)).toEqual(["done", "active"]);
    s = chatReducer(s, { type: "event", event: { type: "token", data: { text: "Hel" } } });
    s = chatReducer(s, { type: "event", event: { type: "token", data: { text: "lo" } } });
    expect(s.text).toBe("Hello");
    s = chatReducer(s, {
      type: "event",
      event: {
        type: "answer",
        data: {
          answer: "Hello [1]",
          refused: false,
          citations: [],
          retrieved_chunk_ids: [],
          steps: [],
          usage: { prompt_tokens: 1, completion_tokens: 1, llm_calls: 1, cost_usd: 0, latency_ms: 1 },
        },
      },
    });
    expect(s.text).toBe("Hello [1]");
    s = chatReducer(s, { type: "event", event: { type: "done", data: {} } });
    expect(s.status).toBe("done");
    expect(s.steps.every((x) => x.status === "done")).toBe(true);
    // Events after completion are ignored.
    expect(chatReducer(s, { type: "event", event: { type: "token", data: { text: "!" } } })).toBe(s);
  });

  it("treats a stream that ends without an answer as an error", () => {
    let s = chatReducer(initialChatState, { type: "start", question: "q" });
    s = chatReducer(s, { type: "event", event: { type: "token", data: { text: "Partial" } } });
    s = chatReducer(s, { type: "event", event: { type: "done", data: {} } });
    expect(s).toMatchObject({ status: "error", error: STREAM_ENDED_EARLY, answer: null, text: "Partial" });
  });

  it("records server errors", () => {
    let s = chatReducer(initialChatState, { type: "start", question: "q" });
    s = chatReducer(s, { type: "event", event: { type: "error", data: { message: "boom" } } });
    expect(s).toMatchObject({ status: "error", error: "boom" });
  });
});

describe("parseHash", () => {
  it("routes hashes, including run deep links with a question", () => {
    expect(parseHash("", "eval")).toEqual({ name: "eval" });
    expect(parseHash("#/", "chat")).toEqual({ name: "chat" });
    expect(parseHash("#/eval/")).toEqual({ name: "eval" });
    expect(parseHash("#/runs/2026-10-28T1402-full?q=fb-00412")).toEqual({
      name: "run",
      runId: "2026-10-28T1402-full",
      qid: "fb-00412",
    });
    expect(parseHash("#/nope")).toEqual({ name: "notFound", path: "/nope" });
  });
});

const result = (over: Partial<QuestionResult>): QuestionResult => ({
  qid: "x",
  dataset: "xbrl",
  question: "What was revenue?",
  answerable: true,
  gold: "1",
  prediction: null,
  scores: {},
  ...over,
});

describe("run results", () => {
  it("derives outcomes from the primary score", () => {
    expect(questionOutcome(result({ scores: { numeric_accuracy: 1 } }))).toBe("correct");
    expect(questionOutcome(result({ scores: { numeric_accuracy: null, judge_accuracy: 0 } }))).toBe("incorrect");
    expect(questionOutcome(result({ scores: { refusal_accuracy: 1 } }))).toBe("correct");
    expect(questionOutcome(result({ error: "Timeout" }))).toBe("error");
    expect(questionOutcome(result({}))).toBe("unscored");
  });

  it("filters by dataset, outcome and search text", () => {
    const rows = [
      result({ qid: "a", dataset: "xbrl", scores: { numeric_accuracy: 1 }, ticker: "AAPL" }),
      result({ qid: "b", dataset: "financebench", scores: { numeric_accuracy: 0 } }),
    ];
    expect(filterResults(rows, { dataset: "xbrl", outcome: "all", search: "" }).map((r) => r.qid)).toEqual(["a"]);
    expect(filterResults(rows, { dataset: "all", outcome: "incorrect", search: "" }).map((r) => r.qid)).toEqual(["b"]);
    expect(filterResults(rows, { dataset: "all", outcome: "all", search: "aapl" }).map((r) => r.qid)).toEqual(["a"]);
  });
});
