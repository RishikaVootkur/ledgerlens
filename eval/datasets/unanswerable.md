# Datasheet: unanswerable set (`unanswerable.jsonl`)

## Motivation

An agent that answers every question will hallucinate on questions its corpus cannot support. This set measures refusal accuracy (M6, M7): the right response to each row is to say the filings do not contain the answer. The questions are plausible and specific, so they test hallucination under pressure and not the rejection of obvious nonsense.

## Composition

60 handwritten questions, 10 per reason category. `answerable` is false and `answer`, `answer_value` are empty on every row. `question_type` is the reason category.

| question_type | dev | test | total | what it asks |
|---|---:|---:|---:|---|
| future_period | 4 | 6 | 10 | results for periods after the newest corpus filing (Q4 2026, FY2027 to FY2030) |
| not_in_corpus_company | 4 | 6 | 10 | real companies outside the universe and outside FinanceBench (Tesla, Oracle, Home Depot, ...) |
| not_disclosed | 4 | 6 | 10 | facts filings never contain (iPhone units after 2019, per-customer revenue, an executive's home address, future stock prices) |
| wrong_form_or_period | 4 | 6 | 10 | quarters from 2008 to 2012, older than any ingested filing for that company |
| false_premise | 4 | 6 | 10 | segments that do not exist or were divested (J&J Consumer Health after Kenvue, 3M Health Care after Solventum, Verizon Media) |
| nonfinancial_trivia | 4 | 6 | 10 | company history and personal trivia the filings do not cover |
| **total** | 24 | 36 | 60 | |

## Collection

Written by hand in `src/ledgerlens/eval/datasets/seeds/unanswerable.yaml`. Each entry has a stable id (`unans-{category}-{nn}`), the category, the question, optional ticker, company and fiscal period, and a `why` note. `ledgerlens datasets build --only unanswerable` validates the seeds (known category, no unknown fields, `not_in_corpus_company` tickers outside the universe) and writes the JSONL.

## Labeling

The label is the category plus the `why` note (copied to `metadata.why`), checked against the corpus definition: the latest 10-K and two latest 10-Qs per universe company as of 2026-11-01, plus the FinanceBench source documents (which ingestion also loads). Old-period questions avoid periods that FinanceBench documents cover, including the multi-year comparatives inside older 10-Ks, by asking about specific quarters from 2008 to 2012. `metadata.in_universe` marks questions about universe companies, which are the harder cases because retrieval returns relevant-looking chunks.

## Splits

Within each category, rows are ordered by the SHA-256 hex digest of the `qid`; the first `round(0.4 * n)` are `dev`, the rest `test`. The test split is held out and reported once, at the end.

## Known limitations

- "Unanswerable" is relative to the corpus snapshot. Re-ingesting later filings can make future_period questions answerable; review this set whenever the corpus changes.
- wrong_form_or_period questions are answerable from historical XBRL. If the agent's `get_company_facts` tool returns facts outside the corpus filings, these rows stop being unanswerable; restrict that tool to corpus accessions or report this category separately.
- A few not_disclosed and false_premise questions sit near topics a filing does discuss (Microsoft's OpenAI relationship, J&J's Kenvue separation). A cautious answer that cites related disclosure and still declines the number is correct.
- 60 rows give wide intervals on refusal accuracy per category (10 each).
- The questions were written by one author; their phrasing may not match how real users ask.

## License

MIT, like the rest of the repo.

## Maintenance

Append new entries to the YAML with new ids; never renumber or reuse ids, so splits stay stable. Rebuild with `uv run ledgerlens datasets build --only unanswerable`.
