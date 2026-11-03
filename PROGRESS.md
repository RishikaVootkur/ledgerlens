# Progress log

Newest entry first. Updated at the end of every PR so the build can resume from here.

## Status

- Merged (17 PRs, each reviewed by a fresh-context subagent with findings fixed before merge): skeleton, shared contracts, LLM provider, metrics/stats/gate, retrieval, datasets, web UI, agent, API (SSE), MCP server, tracing, Docker stack, ingestion, eval runner + judge + CI gate, web metric names, dashboard + Pages + live eval, targeted fixes.
- Results PR (#18): run files, experiments manifest and dashboard, README, resume bullets, re-recorded 30-question CI slice (Flash-Lite, smoke + FinanceBench + unanswerable), held-out test run.
- Blockers needing the user:
  1. `SEC_USER_AGENT` repo secret (the live-eval workflow skips until it exists; `GEMINI_API_KEY` is set).
  2. Phoenix 21 has login on: put a Phoenix system API key in `.env` as `PHOENIX_API_KEY` (with `PHOENIX_ENDPOINT`) to see traces. Tracing is verified against a local OTLP receiver.
- Fix 4 (statement-aware retrieval, D21) came after the first held-out run; the test split was re-run once and both runs are reported.
- Known follow-ups: re-ingest with the improved parser from PR #13 (better section labels for INTC/JPM/LIN, no duplicated PDF table text) and re-run the evals; restrict the XBRL tool to the corpus filings' fiscal years (D20); embed fixed chunks for the FinanceBench test documents.

## Environment facts (checked 2026-11-01 to 11-03)

- Free-tier limits that shape the experiments: Groq 200,000 tokens/day and 8,000 tokens/minute per model (7,000 input tokens per request for Qwen3.8 27B); Gemini 3.8 Flash 20 requests/day; Gemini 3.5 Flash-Lite about 15 requests/minute with a larger daily allowance.
- Gemini is called through Google's OpenAI-compatible endpoint (the native endpoint 404s for new models).
- Ollama: `qwen3:8b`, `nomic-embed-text` (embedding ~5-8 chunks/s on real chunks, capped at 2,000 characters of input).
- Qdrant 1.20.0 and Phoenix 21.0.0 in Docker; MCP Python SDK 2.3.0 on protocol 2026-07-28.
- Index fingerprints: section `446f34acdfa5-62209`, fixed `f969b924674e-8772` (FinanceBench dev documents only).

## Key numbers

- Corpus: 174 filings, 62,209 section chunks, 31,604 fixed chunks (8,772 embedded).
- Judge vs FinanceBench human labels (held-out, n=147): agreement 0.898 [0.838, 0.937], kappa 0.846; clean subset (n=52) 0.865, kappa 0.788.
- Retrieval (FinanceBench dev, n=60): hybrid MRR 0.235 vs BM25 0.167 (Holm p = 0.001).
- Agent, 62-question dev slice, Flash-Lite: numeric 0.812, judge 0.550, refusal 0.871; without XBRL tool numeric 0.656.
- Models on the smoke slice: Groq Qwen3.8 27B 0.95, Flash-Lite 0.95, local Qwen3 8B 0.40.
- Held-out test (120 questions): before fix 4 numeric 0.806 [0.70, 0.88], judge 0.556, unanswerable refusal 24/24, citation precision 1.00, recall 0.91, FinanceBench false refusals 31%; after fix 4 the same except FinanceBench false refusals 28% and recall 0.89.

## Working agreements

- Experiments run from a worktree pointed at the shared `data/` and `.cache/`; every run file records its config and commit.
- Any change to prompts, tools or retrieval invalidates the CI replay cache: re-record with the command at the top of `eval/configs/ci_slice.yaml` and commit `eval/cache` + `eval/baselines/ci_slice.json`.

## Log

### 2026-11-02 to 11-03
- Ran the final ablations on Gemini 3.5 Flash-Lite after Groq's daily token budget ran out; three targeted fixes from error analysis merged (#17). The derived-number fix was redesigned after measuring 99% false acceptance in its first version.
- Scorer bugs found and fixed by replaying cached responses (string values, double scaling, derived-answer citation support).

### 2026-11-01
- Built and merged the platform through reviewed PRs; ingested 174 filings; calibrated the judge (prompt v1 0.75 -> v2 0.90 agreement on dev).
