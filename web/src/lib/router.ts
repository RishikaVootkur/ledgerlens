import { useSyncExternalStore } from "react";

// Hash routing works on GitHub Pages without server rewrites, so deep links
// such as #/runs/abc survive a reload in every mode.

export type Route =
  | { name: "chat" }
  | { name: "eval" }
  | { name: "run"; runId: string; qid: string | null }
  | { name: "notFound"; path: string };

export function parseHash(hash: string, defaultRoute: "chat" | "eval" = "chat"): Route {
  const raw = hash.replace(/^#/, "");
  const [pathPart = "", queryPart = ""] = raw.split("?", 2);
  const path = pathPart.replace(/\/+$/, "") || "/";
  const query = new URLSearchParams(queryPart);
  if (path === "/") return { name: defaultRoute };
  if (path === "/ask") return { name: "chat" };
  if (path === "/eval") return { name: "eval" };
  const run = /^\/runs\/([^/]+)$/.exec(path);
  if (run) return { name: "run", runId: decodeURIComponent(run[1]!), qid: query.get("q") };
  return { name: "notFound", path };
}

export const href = {
  chat: () => "#/ask",
  eval: () => "#/eval",
  run: (runId: string, qid?: string) =>
    `#/runs/${encodeURIComponent(runId)}${qid ? `?q=${encodeURIComponent(qid)}` : ""}`,
};

/** Query parameters after `?` in the hash, e.g. #/ask?q=... */
export function hashQuery(hash: string = window.location.hash): URLSearchParams {
  const i = hash.indexOf("?");
  return new URLSearchParams(i === -1 ? "" : hash.slice(i + 1));
}

function subscribe(callback: () => void): () => void {
  window.addEventListener("hashchange", callback);
  return () => window.removeEventListener("hashchange", callback);
}

export function useHash(): string {
  return useSyncExternalStore(
    subscribe,
    () => window.location.hash,
    () => "",
  );
}

/** Replace the hash without adding a history entry (used for drawer state). */
export function replaceHash(next: string): void {
  const url = `${window.location.pathname}${window.location.search}${next}`;
  window.history.replaceState(null, "", url);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}
