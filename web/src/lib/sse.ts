import type { AgentAnswer, ChatEvent, StepEventData } from "../api/types";

/** One dispatched Server-Sent Event, before JSON decoding. */
export interface RawSSEEvent {
  event: string;
  data: string;
  id?: string;
}

export interface SSEParser {
  /** Feed the next decoded text chunk; chunks may split lines anywhere. */
  feed(chunk: string): void;
  /** Dispatch a trailing event that was not followed by a blank line. */
  flush(): void;
}

/**
 * Incremental parser for the text/event-stream format
 * (https://html.spec.whatwg.org/multipage/server-sent-events.html).
 * Needed because /api/chat is a POST, which EventSource cannot send.
 */
export function createSSEParser(onEvent: (event: RawSSEEvent) => void): SSEParser {
  let buffer = "";
  let eventName = "";
  let dataLines: string[] = [];
  let lastId: string | undefined;
  let pendingCR = false;

  const dispatch = () => {
    if (dataLines.length > 0) {
      onEvent({ event: eventName || "message", data: dataLines.join("\n"), id: lastId });
    }
    eventName = "";
    dataLines = [];
  };

  const processLine = (line: string) => {
    if (line === "") {
      dispatch();
      return;
    }
    if (line.startsWith(":")) return; // comment / keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    switch (field) {
      case "event":
        eventName = value;
        break;
      case "data":
        dataLines.push(value);
        break;
      case "id":
        if (!value.includes("\0")) lastId = value;
        break;
      default:
        break; // retry and unknown fields are ignored
    }
  };

  return {
    feed(chunk: string) {
      let text = chunk;
      // A "\r\n" pair may be split across chunks.
      if (pendingCR && text.startsWith("\n")) text = text.slice(1);
      pendingCR = false;
      buffer += text;
      let start = 0;
      for (let i = 0; i < buffer.length; i++) {
        const ch = buffer[i];
        if (ch === "\n" || ch === "\r") {
          processLine(buffer.slice(start, i));
          if (ch === "\r") {
            if (i + 1 < buffer.length) {
              if (buffer[i + 1] === "\n") i++;
            } else {
              pendingCR = true;
            }
          }
          start = i + 1;
        }
      }
      buffer = buffer.slice(start);
    },
    flush() {
      if (buffer.length > 0) {
        processLine(buffer);
        buffer = "";
      }
      dispatch();
    },
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Decode a raw SSE event into a typed chat event; unknown or malformed events return null. */
export function toChatEvent(raw: RawSSEEvent): ChatEvent | null {
  let payload: unknown;
  try {
    payload = raw.data === "" ? {} : JSON.parse(raw.data);
  } catch {
    return null;
  }
  if (!isRecord(payload)) return null;
  switch (raw.event) {
    case "step":
      if (typeof payload.name !== "string") return null;
      return {
        type: "step",
        data: {
          name: payload.name,
          detail: typeof payload.detail === "string" ? payload.detail : "",
        } satisfies StepEventData,
      };
    case "token":
      return typeof payload.text === "string" ? { type: "token", data: { text: payload.text } } : null;
    case "answer":
      return typeof payload.answer === "string"
        ? { type: "answer", data: normalizeAnswer(payload) }
        : null;
    case "error":
      return {
        type: "error",
        data: { message: typeof payload.message === "string" ? payload.message : "Unknown error" },
      };
    case "done":
      return { type: "done", data: {} };
    default:
      return null;
  }
}

/** Fill defaults the Pydantic model would have, so the UI never sees undefined arrays. */
function normalizeAnswer(payload: Record<string, unknown>): AgentAnswer {
  const answer = payload as Partial<AgentAnswer>;
  return {
    answer: answer.answer ?? "",
    refused: answer.refused ?? false,
    answer_value: answer.answer_value ?? null,
    answer_unit: answer.answer_unit ?? null,
    citations: answer.citations ?? [],
    retrieved_chunk_ids: answer.retrieved_chunk_ids ?? [],
    steps: answer.steps ?? [],
    usage: {
      prompt_tokens: answer.usage?.prompt_tokens ?? 0,
      completion_tokens: answer.usage?.completion_tokens ?? 0,
      llm_calls: answer.usage?.llm_calls ?? 0,
      cost_usd: answer.usage?.cost_usd ?? 0,
      latency_ms: answer.usage?.latency_ms ?? 0,
    },
  };
}
