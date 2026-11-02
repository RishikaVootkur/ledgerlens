# Progress log

Newest entry first. Updated at the end of every PR so the build can resume from here.

## Status

- Merged: #1 skeleton, #2 contracts, #3 LLM provider, #4 metrics/stats/gate, #5 retrieval, #6 datasets, #7 web UI, #8 agent, #10 MCP server.
- In review: #9 API (SSE), #11 tracing, #12 Docker stack (stacked, builder fixing review items), #13 ingestion, #14 eval runner + judge + CI gate, dashboard export + Pages + live eval (stacked on #14), web metric-name alignment.
- Running: full embedding of 62,209 section chunks, then 19,105 FinanceBench fixed chunks (~8.5 chunks/s).
- Next: retrieval ablation (8 configs, LLM-free), agent ablations on the 62-question slice, model comparison, error taxonomy and two targeted fixes, held-out test run, README and resume bullets.
- Blockers needing the user:
  1. Repo secrets: `gh secret set GEMINI_API_KEY` and `gh secret set SEC_USER_AGENT` were blocked by the permission system. The live-eval workflow skips cleanly until they exist.
  2. Phoenix 21 has login on by default, so traces need a Phoenix system API key in `.env` (`PHOENIX_API_KEY`). Creating one via the default admin login was blocked by the permission system. Tracing is verified against a local OTLP receiver (17 spans per question with tokens and cost).

## Environment facts (checked 2026-11-01)

- Gemini free tier: `gemini-3.8-flash` allows only 20 requests/day; `gemini-3.5-flash-lite` works with more headroom. Called through Google's OpenAI-compatible endpoint (the native endpoint 404s for new models).
- Groq free tier: 1000 requests/day, 8000 tokens/minute per model, and 7000 input tokens per single request for Qwen3.8 27B.
- Ollama: `qwen3:8b`, `nomic-embed-text` (8.5 chunks/s on real section chunks).
- Docker: 8 GB for containers; Qdrant 1.20.0, Phoenix 21.0.0.
- MCP Python SDK 2.3.0, protocol revision 2026-07-28 (`server/discover` replaces `initialize`; legacy clients negotiate 2025-11-25).

## Key numbers

- Corpus: 174 filings (30 companies x 10-K + two 10-Qs, plus 84 FinanceBench PDFs), 203,956 blocks, 14,306 tables (0 broken), 62,209 section chunks, 31,604 fixed chunks.
- Datasets: FinanceBench 150, XBRL 350 (0 gold mismatches on independent re-derivation), unanswerable 60, smoke 20, judge calibration 240.
- CI slice baseline (20 XBRL questions, Qwen3.8 27B): numeric accuracy 0.80 [0.58, 0.92], refusal accuracy 0.95, citation precision 1.00, MRR 0.875, model latency p50 3.3 s, $0.008/question list-price equivalent.
- Judge (GPT-OSS 120B, prompt v2) vs FinanceBench human labels, held-out n=144: agreement 0.896 [0.835, 0.936], kappa 0.844 [0.769, 0.907], error rate 0.104.
- Retrieval, FinanceBench dev, BM25 within document: recall@10 0.367, MRR 0.167.

## Working agreements

- Parallel builders work in `.claude/worktrees/` (ignored by git and ruff). Shared contracts live in `src/ledgerlens/schemas.py`.
- Every PR gets a fresh-context reviewer subagent; findings are fixed on the same branch before merge.
- Any agent or prompt change invalidates the CI replay cache: re-record with `ledgerlens eval run --config eval/configs/ci_slice.yaml --split dev --run-id ci-slice-baseline --out eval/baselines/ci_slice.json --record-cache` and commit `eval/cache` + the baseline.

## Log

### 2026-11-01 (evening)
- Smoke slice exposed three issues, all fixed: answer prompts above Groq's per-request cap, wall latency dominated by rate-limit waits (now reported separately as model latency), XBRL facts not counted as retrieved evidence.
- Judge prompt v1 -> v2 on the calibration dev split (agreement 0.75 -> 0.91) by encoding how annotators label refusals; held-out numbers above.
- Agent review (PR #8) found six blockers (self-check bypasses, unchecked small numbers, calculate laundering invented inputs, pre-split EPS); all fixed with regression tests.
- Switched default answer model to Qwen3.8 27B on Groq after hitting Gemini 3.8 Flash's 20/day limit.

### 2026-11-01
- Created the public repo; skeleton, contracts, provider, metrics, retrieval, datasets, web UI, agent and MCP server merged through reviewed PRs.
