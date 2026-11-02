import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, streamChat } from "./client";
import type { ChatEvent } from "./types";

function stubFetch(body: string, init: ResponseInit) {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(body, init)));
}

afterEach(() => vi.unstubAllGlobals());

describe("streamChat", () => {
  it("streams typed events from a text/event-stream response", async () => {
    stubFetch('event: token\ndata: {"text":"Hi"}\n\nevent: done\ndata: {}\n\n', {
      headers: { "content-type": "text/event-stream; charset=utf-8" },
    });
    const events: ChatEvent[] = [];
    await streamChat({ question: "q" }, (e) => events.push(e));
    expect(events.map((e) => e.type)).toEqual(["token", "done"]);
  });

  it("rejects a 200 response that is not an event stream", async () => {
    stubFetch("<!doctype html><title>Not found</title>", { headers: { "content-type": "text/html" } });
    const onEvent = vi.fn();
    await expect(streamChat({ question: "q" }, onEvent)).rejects.toThrow(/instead of text\/event-stream/);
    expect(onEvent).not.toHaveBeenCalled();
  });

  it("rejects HTTP errors with the status code", async () => {
    stubFetch("upstream timeout", { status: 502, statusText: "Bad Gateway" });
    const err = await streamChat({ question: "q" }, vi.fn()).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(502);
  });
});
