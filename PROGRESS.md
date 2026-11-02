# Progress log

Newest entry first. Updated at the end of every PR so the build can resume from here.

## Status

- Done (merged): #1 skeleton, #2 shared deps and contracts, #3 LLM provider layer, #4 eval metrics/stats/compare/gate, #5 hybrid retrieval.
- In review: #6 eval datasets (fixing review findings), #7 web UI (fixing review findings), agent (`feat/agent`, PR next).
- In progress: ingestion (`feat/edgar-ingestion`, builder agent running the full corpus), slice index build.
- Next: eval runner + `eval run` / `eval compare` CLI, CI regression gate with committed replay cache, MCP server, FastAPI SSE, observability, experiments.
- Blockers: none.

## Environment facts (checked 2026-11-01)

- Gemini free tier on this key: `gemini-3.8-flash` and `gemini-3.5-flash-lite` answer. `gemini-4-argon` and `gemini-3.1-pro-preview` have zero free quota. The native `generateContent` endpoint returns 404 for new models, so Gemini is called through Google's OpenAI-compatible endpoint. It sometimes returns "high demand" 503s; the provider retries them.
- Groq free tier: `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`. Limits: 1000 requests/day, 8000 tokens/minute per model.
- Ollama: `qwen3:8b`, `nomic-embed-text`. Embedding runs at ~19 chunks/s on short chunks, ~4.5 chunks/s on the real section chunks while other jobs share the CPU.
- Docker Desktop has 8 GB for containers. Qdrant 1.20.0 and Phoenix 21.0.0 run from `docker-compose.yml`.
- MCP Python SDK 2.3.0 speaks protocol revision 2026-07-28.

## Key numbers

- Corpus so far: 90 EDGAR filings (30 companies x latest 10-K + two 10-Qs), 25,423 section chunks and 12,668 fixed chunks; FinanceBench PDFs in progress.
- Datasets (PR #6): FinanceBench 150, XBRL 350 (0 gold mismatches on independent re-derivation), unanswerable 60, smoke 20.

## Working agreements

- Parallel builders work in `.claude/worktrees/` (ignored by git and ruff). Shared contracts live in `src/ledgerlens/schemas.py`; dependencies were added up front so branches never conflict on `uv.lock`.
- Every PR gets a fresh-context reviewer subagent whose findings are posted as a PR comment; findings are fixed on the same branch before merge.

## Log

### 2026-11-01
- Merged #3 provider layer after review fixes (concurrent cache writes, retry allow-list, daily quota detection, token pacing, replay isolation).
- Merged #5 retrieval after review fixes (thread-safe SQLite, stale chunk removal, Qdrant resync).
- Merged #4 metrics after review fixes ("3M" no longer parsed as 3 million, no forgiveness of 100x percent errors, Wilson intervals for binary metrics, gate fails when a candidate scores fewer questions).
- Agent built: LangGraph gather/tools loop, JSON answer with [E#] evidence refs renumbered to [n], deterministic number grounding and citation self-check, one repair, refusal when nothing supports the answer.
- Created the public repo with the scaffold commit. Started `chore/project-skeleton`.
