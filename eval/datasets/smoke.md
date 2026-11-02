# Datasheet: smoke set (`smoke.jsonl`)

## Motivation

A small, fast, fixed set for the thin end-to-end slice (M2) and the CI regression gate, which replays a committed response cache on every PR. It must run in minutes and only touch the three slice companies that the first ingestion covers.

## Composition

20 questions copied from the XBRL golden set, all about AAPL, JPM and KO (`SLICE_TICKERS`), all in `dev`.

| question_type | dev | test | total |
|---|---:|---:|---:|
| multi_step | 4 | 0 | 4 |
| single_fact | 12 | 0 | 12 |
| yoy_change | 4 | 0 | 4 |
| **total** | 20 | 0 | 20 |

By company: 8 AAPL, 6 JPM, 6 KO. Questions span the FY2025 10-Ks and the 2026 10-Qs.

## Collection

Built from `xbrl.jsonl` by `src/ledgerlens/eval/datasets/smoke.py`. Rebuilding xbrl always rebuilds smoke so the two stay in sync.

## Labeling

Labels, tolerances, evidence and metadata are the XBRL rows' own (see `xbrl.md`). Each smoke row has `qid` `smoke-...` mirroring its source (`xbrl-...`) and keeps the source id in `metadata.source_qid`.

## Splits

The pool is XBRL questions about the slice companies that are already in the XBRL `dev` split, so no held-out XBRL test question leaks into a set that is tuned against. From that pool, per type, questions are taken round-robin over AAPL, JPM, KO in SHA-256 order of `"smoke:" + source qid` until the mix (12 single_fact, 4 yoy_change, 4 multi_step) is filled. Every smoke row is `dev`.

## Known limitations

- 20 numeric questions about three companies say little about overall quality; use it to catch regressions, not to compare systems.
- No FinanceBench, free-text or unanswerable questions, so the gate does not cover judge scoring or refusals.
- Overlaps the XBRL dev split by construction; do not add smoke scores to XBRL dev scores.

## License

MIT, derived from public SEC EDGAR data.

## Maintenance

Regenerate with `uv run ledgerlens datasets build --only smoke` (or `--only xbrl`). Changing it invalidates the committed CI response cache, which must be re-recorded in the same PR.
