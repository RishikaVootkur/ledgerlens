const env = import.meta.env;

/** Static build for GitHub Pages: no backend, chat is disabled. */
export const IS_STATIC = env.VITE_STATIC === "1";

/** Replay a canned SSE sequence instead of calling /api/chat. */
export const MOCK_CHAT = env.VITE_MOCK_CHAT === "1";

/** Where the FastAPI routes live. */
export const API_BASE: string = env.VITE_API_BASE ?? "/api";

/** Where dashboard.json and runs/{run_id}.json live. */
export const DATA_BASE: string = (env.VITE_DATA_BASE ?? "/api/data").replace(/\/$/, "");

export const REPO_URL = "https://github.com/RishikaVootkur/ledgerlens";
export const RUN_LOCALLY_URL = `${REPO_URL}#readme`;
