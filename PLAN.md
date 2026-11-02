# LedgerLens build plan

Milestones in build order. Each one lands as one or more PRs. A milestone is done when every acceptance criterion holds on `main` and CI is green.

## M0. Skeleton
- `uv` project on Python 3.12, typed config (`pydantic-settings`), structured logging (`structlog`), Typer CLI.
- ruff, mypy (strict), pytest, pre-commit with gitleaks. CI runs lint, types, tests, gitleaks on every PR.
- Accept: `make test` passes locally and in CI; `.env` is ignored; `.env.example` exists.

## M1. LLM provider layer
- One entry point for every LLM call (LiteLLM) covering Gemini (OpenAI-compatible endpoint), Groq and Ollama.
- Disk cache keyed on the full request, retries with exponential backoff on 429/5xx, usage and list-price-equivalent cost on every call.
- Accept: unit tests for cache keys, cost math and retry behavior; a cached call is free and identical on rerun.

## M2. Thin end-to-end slice
- Ingest 3 companies from EDGAR (latest 10-K and two 10-Qs), parse HTML into sections with Markdown tables, chunk, quality report.
- Qdrant (Docker) dense index with `nomic-embed-text`, BM25, reciprocal rank fusion, metadata filters.
- LangGraph agent with `search_filings`, `get_company_facts`, `calculate`; answers with inline citations.
- 20-question XBRL eval with numeric scoring, run files, `ledgerlens eval run` / `eval compare`.
- CI regression gate replays the committed response cache and comments a metric diff on the PR.
- Accept: a fresh clone can run the gate with no keys; the agent answers a question with a correct citation.

## M3. Ingestion at scale
- About 30 companies across sectors plus every filing the FinanceBench open sample needs (PDF parsing with page numbers).
- Idempotent and resumable (manifest of fetched and parsed files), SEC rate limit under 10 requests per second.
- Data quality report: empty sections, broken tables, duplicate chunks.
- Accept: `make ingest` twice in a row does no new work the second time; quality report committed.

## M4. Evaluation datasets
- FinanceBench open sample (150), XBRL golden set (200+, including multi-step arithmetic), unanswerable set (50+).
- Fixed dev and test splits, a datasheet for each dataset, license and credit for FinanceBench.
- Accept: datasets and datasheets committed; split assignment deterministic and documented.

## M5. Retrieval depth
- Two chunking strategies (fixed size, section and table aware), optional cross-encoder rerank.
- Retrieval metrics: recall@k, MRR, nDCG against labeled evidence, with unit tests against hand-computed values.
- Accept: ablation table (BM25, dense, hybrid, hybrid plus rerank; both chunkers) with bootstrap intervals.

## M6. Full agent
- Plan, retrieve, verify numbers against XBRL, answer with citations (filing, section, page), self-check that each claim has a supporting chunk, refuse when unsupported.
- `get_table` tool. Safe `calculate` that parses arithmetic only.
- Accept: unit tests for tools and the calculator; refusal accuracy measured on the unanswerable set.

## M7. Judge and statistics
- LLM judge from a different model family than the answer model; calibration against FinanceBench human labels (accuracy, Cohen's kappa, error rate).
- Citation precision and recall, refusal accuracy, latency p50/p95, tokens, cost.
- Bootstrap 95% intervals on headline metrics; paired tests (McNemar for binary, paired bootstrap / permutation for continuous) in `eval compare`.
- Accept: calibration report committed; `eval compare` prints deltas with p-values.

## M8. MCP server
- Same tools exposed through the official MCP Python SDK on the current spec revision.
- Accept: a test drives the server from an MCP client session and calls every tool.

## M9. API, UI, observability
- FastAPI with SSE streaming answers. React (Vite, TypeScript, Tailwind) chat with citations panel, and eval dashboard (runs, metrics with intervals, ablations, model comparison, failures).
- OpenTelemetry + OpenInference into Arize Phoenix: one span per agent step and tool call with tokens, latency, cost.
- `docker compose up` brings up Qdrant, Phoenix, API and UI.
- Accept: compose stack comes up; a trace with per-step spans appears in Phoenix.

## M10. Experiments and error analysis
- Ablations: retrieval methods, chunkers, XBRL verification on/off, three models (Gemini, Groq, local qwen3:8b).
- Failure taxonomy with counts and examples; at least two targeted fixes with before and after numbers.
- Final held-out test-set run reported once.
- Accept: `docs/results.md` with every table backed by a committed run file.

## M11. CI and publishing
- Scheduled live eval on a small slice with the `GEMINI_API_KEY` secret.
- Static eval dashboard published to GitHub Pages from the latest results.
- Accept: both workflows green; Pages URL live.

## M12. Docs
- Recruiter README (pitch, screenshots, Mermaid diagram, results tables, five-command quickstart, licenses, what I learned).
- `docs/decisions.md`, `docs/eval-methodology.md`, datasheets, `RESUME_BULLETS.md` linked to run files.
