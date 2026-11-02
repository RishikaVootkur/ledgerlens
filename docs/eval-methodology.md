# Evaluation methodology

How LedgerLens is measured, and why the numbers can be trusted. Every number in the README comes from a run file in `eval/runs/` produced by the commands below.

## Datasets

| Dataset | Rows | Ground truth | Used for |
|---|---:|---|---|
| FinanceBench open sample | 150 | Human-written answers and evidence pages (Patronus AI, CC BY-NC 4.0) | Free-text and numeric answers on messy PDFs, judge calibration, page-level retrieval |
| XBRL golden set | 350 | Values the filer tagged in XBRL (SEC companyfacts), matched by accession number and period | Numeric accuracy, including year-over-year and two-fact arithmetic |
| Unanswerable | 60 | Questions the corpus cannot answer (future periods, uncovered companies, undisclosed facts, false premises) | Refusal accuracy and hallucination rate |
| Smoke | 20 | A dev-only subset of XBRL for AAPL, JPM and KO | The CI regression gate |

Each dataset has a datasheet in `eval/datasets/`. Splits are fixed by a hash of the question id within each question type (about 40% dev, 60% test). The test split is held out: it is run once, at the end, and only reported.

## Running an experiment

```bash
uv run ledgerlens eval run --config eval/configs/default.yaml --split dev
uv run ledgerlens eval compare <baseline-run> <candidate-run>
uv run ledgerlens eval retrieval --config eval/configs/<retrieval-config>.yaml --split dev
```

A config (`eval/configs/*.yaml`) holds every setting that can change an answer: model, chunking strategy, retrieval method and k, agent rounds, whether the XBRL tool, number verification and the citation self-check are on, the datasets and the search scope. The run file stores the full config, the git commit, per-question predictions with citations, usage and scores, and summaries with 95% intervals.

## Metrics

Per question (keys in the run file match `eval/gate.yaml`):

- `numeric_correct`: the answer's number is within the question's tolerance of the gold (relative 1% for reported facts, 2% for derived ratios, plus an absolute tolerance for values near zero). Scale errors (millions vs billions) are never forgiven.
- `refusal_correct`: the agent refused exactly when the question is unanswerable. On answerable questions this measures false refusals; on the unanswerable set it measures hallucinations.
- `citation_precision`: share of an answer's citations that support it. For numeric answers a citation supports the answer when its source states the headline number at some reporting scale; for text answers when its quote appears in the source. The LLM judge adds a semantic support check for text answers.
- `citation_recall`: share of the labeled evidence (FinanceBench pages, or the XBRL fact's filing) that the answer cites.
- Retrieval, scored against labeled evidence: `recall_at_k`, `mrr`, `ndcg_at_10`. FinanceBench evidence is page-level; XBRL evidence is filing-level, so XBRL retrieval scores are coarse and the retrieval ablation uses FinanceBench.
- Operations: end-to-end latency p50 and p95, tokens and list-price-equivalent cost per question.

Every metric function is unit-tested against hand-computed values (`tests/eval/`).

## Statistics

- Binary metrics get Wilson score intervals; continuous metrics get 2,000-sample percentile bootstrap intervals (seeded, so summaries are reproducible).
- `eval compare` aligns two runs on their common questions and tests the paired difference: exact McNemar for binary metrics, a sign-flip permutation test for continuous ones, with Holm correction across metrics.

## Determinism and the CI regression gate

Every LLM response and every tool result (search hits, XBRL facts, tables) is cached on disk, keyed on a SHA-256 of the full request, including the retrieval configuration and an index fingerprint. Reruns are free and identical.

`--record-cache` copies every cache entry a run touched into `eval/cache/`, which is committed. In CI the gate (`.github/workflows/eval-gate.yml`) sets `LLM_MODE=replay`, so the agent can only read the committed cache: no keys, no network, no Qdrant. It replays the 20-question slice (`eval/configs/ci_slice.yaml`), compares it with the committed baseline (`eval/baselines/ci_slice.json`) and fails if any gated metric drops past its threshold, a question errors, or a metric is scored on fewer questions than before. The metric diff is posted as a PR comment.

A change that alters prompts, tools or retrieval produces new requests, which miss the committed cache and fail the gate loudly. That is intended: the author re-records the slice locally (`eval run --config eval/configs/ci_slice.yaml --record-cache`), reviews the score change, and commits the new baseline and cache with the PR. The slice config pins the index fingerprint it was recorded against (`index_version`), so reindexing the corpus does not silently change replayed search results.
