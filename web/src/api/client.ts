import { API_BASE, DATA_BASE, MOCK_CHAT } from "../lib/env";
import { createSSEParser, toChatEvent } from "../lib/sse";
import type {
  ChatEvent,
  ChatRequest,
  Chunk,
  CompaniesResponse,
  Dashboard,
  ModelsResponse,
  RunFile,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function getJSON<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { headers: { Accept: "application/json" }, signal });
  if (!res.ok) {
    throw new ApiError(`${res.status} ${res.statusText || "error"} for ${url}`, res.status);
  }
  return (await res.json()) as T;
}

// ---------------------------------------------------------------- live API

export async function getModels(signal?: AbortSignal): Promise<ModelsResponse> {
  if (MOCK_CHAT) return (await import("./mockChat")).MOCK_MODELS;
  return getJSON<ModelsResponse>(`${API_BASE}/models`, signal);
}

export async function getCompanies(signal?: AbortSignal): Promise<CompaniesResponse> {
  if (MOCK_CHAT) return (await import("./mockChat")).MOCK_COMPANIES;
  return getJSON<CompaniesResponse>(`${API_BASE}/companies`, signal);
}

const chunkCache = new Map<string, Promise<Chunk>>();

export function getChunk(chunkId: string): Promise<Chunk> {
  const cached = chunkCache.get(chunkId);
  if (cached) return cached;
  const request = MOCK_CHAT
    ? import("./mockChat").then((m) => m.getMockChunk(chunkId))
    : getJSON<Chunk>(`${API_BASE}/chunks/${encodeURIComponent(chunkId)}`);
  // Do not cache failures, so a retry can succeed.
  request.catch(() => chunkCache.delete(chunkId));
  chunkCache.set(chunkId, request);
  return request;
}

/**
 * POST /api/chat and stream the SSE response. Resolves when the stream ends.
 * EventSource cannot POST, so this reads the body with fetch + ReadableStream.
 */
export async function streamChat(
  request: ChatRequest,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  if (MOCK_CHAT) {
    const { replayMockChat } = await import("./mockChat");
    return replayMockChat(request, onEvent, signal);
  }
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(request),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = await res.text();
    } catch {
      // keep statusText
    }
    throw new ApiError(`Chat request failed (${res.status}): ${detail.slice(0, 300)}`, res.status);
  }
  // A static host or proxy may answer with a 200 HTML page; never parse that as a stream.
  const contentType = res.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().startsWith("text/event-stream")) {
    await res.body.cancel().catch(() => undefined);
    throw new ApiError(
      `Chat endpoint returned ${contentType || "no content type"} instead of text/event-stream. Is the API running?`,
      res.status,
    );
  }
  const parser = createSSEParser((raw) => {
    const event = toChatEvent(raw);
    if (event) onEvent(event);
  });
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      parser.feed(value);
    }
    parser.flush();
  } finally {
    reader.releaseLock();
  }
}

// ---------------------------------------------------------------- eval data

export function getDashboard(signal?: AbortSignal): Promise<Dashboard> {
  return getJSON<Dashboard>(`${DATA_BASE}/dashboard.json`, signal);
}

export function getRun(runId: string, signal?: AbortSignal): Promise<RunFile> {
  return getJSON<RunFile>(`${DATA_BASE}/runs/${encodeURIComponent(runId)}.json`, signal);
}
