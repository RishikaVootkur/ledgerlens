import { describe, expect, it } from "vitest";
import { createSSEParser, toChatEvent, type RawSSEEvent } from "./sse";

function parseAll(chunks: string[]): RawSSEEvent[] {
  const events: RawSSEEvent[] = [];
  const parser = createSSEParser((e) => events.push(e));
  for (const c of chunks) parser.feed(c);
  parser.flush();
  return events;
}

describe("createSSEParser", () => {
  it("parses named events with JSON data", () => {
    const events = parseAll([
      'event: step\ndata: {"name":"plan","detail":"x"}\n\n',
      'event: token\ndata: {"text":"Hi"}\n\n',
    ]);
    expect(events).toEqual([
      { event: "step", data: '{"name":"plan","detail":"x"}', id: undefined },
      { event: "token", data: '{"text":"Hi"}', id: undefined },
    ]);
  });

  it("handles events split across arbitrary chunk boundaries", () => {
    const stream = 'event: token\ndata: {"text":"Hello "}\n\nevent: token\ndata: {"text":"world"}\n\n';
    const oneByOne = parseAll(stream.split(""));
    expect(oneByOne.map((e) => e.data)).toEqual(['{"text":"Hello "}', '{"text":"world"}']);
  });

  it("accepts CRLF and CR line endings, including a CRLF split between chunks", () => {
    const events = parseAll(["event: done\r", "\ndata: {}\r\n\r\n", "event: token\rdata: {\"text\":\"a\"}\r\r"]);
    expect(events.map((e) => e.event)).toEqual(["done", "token"]);
  });

  it("joins multi-line data, ignores comments and defaults the event name", () => {
    const events = parseAll([": keep-alive\n", "data: line1\ndata: line2\nid: 7\n\n"]);
    expect(events).toEqual([{ event: "message", data: "line1\nline2", id: "7" }]);
  });

  it("dispatches a trailing event on flush and skips events without data", () => {
    const events = parseAll(["event: ping\n\n", 'event: done\ndata: {}']);
    expect(events).toEqual([{ event: "done", data: "{}", id: undefined }]);
  });
});

describe("toChatEvent", () => {
  it("decodes known events", () => {
    expect(toChatEvent({ event: "step", data: '{"name":"verify","detail":"ok"}' })).toEqual({
      type: "step",
      data: { name: "verify", detail: "ok" },
    });
    expect(toChatEvent({ event: "done", data: "{}" })).toEqual({ type: "done", data: {} });
    expect(toChatEvent({ event: "error", data: '{"message":"boom"}' })).toEqual({
      type: "error",
      data: { message: "boom" },
    });
  });

  it("fills AgentAnswer defaults", () => {
    const ev = toChatEvent({ event: "answer", data: '{"answer":"No.","refused":true}' });
    expect(ev?.type).toBe("answer");
    if (ev?.type === "answer") {
      expect(ev.data.citations).toEqual([]);
      expect(ev.data.refused).toBe(true);
      expect(ev.data.usage.cost_usd).toBe(0);
    }
  });

  it("returns null for malformed or unknown events", () => {
    expect(toChatEvent({ event: "token", data: "not json" })).toBeNull();
    expect(toChatEvent({ event: "token", data: '{"txt":"x"}' })).toBeNull();
    expect(toChatEvent({ event: "mystery", data: "{}" })).toBeNull();
  });
});
