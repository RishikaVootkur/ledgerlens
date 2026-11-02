# Datasheet: judge calibration set

## Motivation
LedgerLens grades free-text FinanceBench answers with an LLM judge. A judge is only useful if it agrees with people, so this set measures that agreement against human labels before any judge-based number is reported.

## Composition
240 items, each a model's answer to a FinanceBench open-sample question, with the label FinanceBench's human annotators gave it: `CORRECT` (80), `INCORRECT` (80) or `REFUSAL` (80). Fields: `cid`, `split`, `financebench_id`, `source_model`, `eval_mode`, `question`, `gold_answer`, `model_answer`, `human_label`.

Answers come from nine published FinanceBench result files covering GPT-4, GPT-4 Turbo (1106 preview), Claude 2 and Llama 2 in closed-book, single-store, shared-store, in-context and oracle settings, so the judge sees many answer styles and error types rather than one model's habits.

## Collection
Downloaded from `results/*.jsonl` in github.com/patronus-ai/financebench (raw copies cached under `data/raw/financebench_results/`). Built by `ledgerlens.eval.datasets.judge_calibration.build()`; the builder is deterministic.

## Labeling
Labels are FinanceBench's human labels, unmodified: "Correct Answer", "Incorrect Answer" and "Refusal" map to `CORRECT`, `INCORRECT` and `REFUSAL`. No LedgerLens author relabeled anything.

## Sampling and splits
Within each label, candidate items are ordered by SHA-256 of their id and the first 80 are kept, so the set is balanced across labels. Within each label, a second hash assigns 40% to `dev` (96 items) and 60% to `test` (144 items). The judge prompt may be iterated on `dev` only; `test` is reported once.

## Known limitations
- Balanced sampling differs from the natural label mix in an eval run, so accuracy here is not the accuracy the judge would have on any one run's distribution. Cohen's kappa and per-label recall are reported alongside accuracy for that reason.
- The same FinanceBench question can appear more than once with answers from different models.
- The labels describe answers from 2023-era models, which fail differently from current models.

## License
FinanceBench (questions, gold answers, model answers and human labels) is © Patronus AI, released under CC BY-NC 4.0. Credit: Islam et al., 2023, "FinanceBench: A New Benchmark for Financial Question Answering". LedgerLens is a non-commercial portfolio project and redistributes this subset under the same license.

## Maintenance
Rebuild with `uv run python -c "from ledgerlens.eval.datasets import judge_calibration as j; j.write(j.build())"`. Rebuilding from the cached raw files is deterministic.
