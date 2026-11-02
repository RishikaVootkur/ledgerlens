# Datasheet: FinanceBench open sample (`financebench.jsonl`)

## Motivation

FinanceBench is a public benchmark for open-book question answering over company filings. LedgerLens uses its 150-question open sample as an external yardstick, so results can be compared with published numbers and with human labels (the judge calibration in M7 uses it).

Credit: Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, Bertie Vidgen. 2023. "FinanceBench: A New Benchmark for Financial Question Answering". arXiv:2311.11944. Data by Patronus AI.

## Composition

150 questions about 32 US public companies, each tied to one source document (`doc_name`, e.g. `3M_2018_10K`): 112 from 10-Ks, 15 from 10-Qs, 14 from earnings releases and 9 from 8-Ks, covering fiscal years 2015 to 2024. Each row has a gold answer, a justification for some rows, and one to three evidence passages with page numbers.

| question_type | dev | test | total |
|---|---:|---:|---:|
| domain_relevant | 20 | 30 | 50 |
| metrics_generated | 20 | 30 | 50 |
| novel_generated | 20 | 30 | 50 |
| **total** | 60 | 90 | 150 |

Answer units: 98 free text, 19 USD, 18 ratio, 15 percent.

## Collection

`ledgerlens datasets build --only financebench` downloads `financebench_merged.jsonl` from the Hugging Face dataset `PatronusAI/financebench`, pinned to revision `e04404e3a97f69f79c14d42f24981a1c9c3bcd18`, into `.cache/huggingface`. Source PDFs live at `https://github.com/patronus-ai/financebench/tree/main/pdfs` and are fetched by ingestion (M3), not by this generator.

## Labeling and mapping

The gold labels are FinanceBench's own. The mapping to `Question` is mechanical:

- `qid` is `fb-{number}` from `financebench_id`; `question_type` is FinanceBench's `question_type` with dashes replaced by underscores.
- `fiscal_year` comes from `doc_period`, `form` from `doc_type`, and `fiscal_period` from the doc name (`FY` for 10-Ks, `Q2` for `..._2023Q2_10Q`, empty for 8-Ks).
- `ticker` is filled from a hand-written company to ticker map.
- `answer_value` is set only when the whole gold answer is a single number. `$` and the question's "in USD millions" or "in USD billions" give a USD value in base units (`$1577.00` in millions is 1,577,000,000). `%` gives a percent stored as written (`1.9%` is 1.9). A bare number is a ratio unless the question asks for USD or a percent. Everything else has `answer_unit: text` and no value.
- `abs_tolerance` is half a unit in the last decimal place of the gold (0.005 for `0.66`, 0.05 for `1.9%`), because FinanceBench golds are rounded. `rel_tolerance` stays at 0.01.
- `question_reasoning`, `domain_question_num`, `justification`, `gics_sector`, `doc_link` and the other source fields are kept in `metadata`.

### Page numbers

FinanceBench's `evidence_page_num` is 0-based. The dataset card does not say so, so it was checked by reading the PDFs with pymupdf and comparing word overlap between `evidence_text` and every page:

| financebench_id | document | label | best PDF page index (0-based) | overlap there | overlap at label - 1 |
|---|---|---:|---:|---:|---:|
| 03029 | 3M_2018_10K | 59 | 59 | 1.00 | 0.31 |
| 04672 | 3M_2018_10K | 57 | 57 | 1.00 | 0.19 |
| 00222 | AMD_2022_10K | 55 | 55 | 1.00 | 0.10 |
| 01351 | AMERICANEXPRESS_2022_10K | 43 | 43 | 1.00 | 0.13 |

Six labels are 0 as well. `Evidence.page` is therefore `evidence_page_num + 1` (1-based, matching the contract), and each row records this in `metadata.evidence_page_base`.

## Splits

Within each `question_type`, rows are ordered by the SHA-256 hex digest of the `qid`; the first `round(0.4 * n)` are `dev`, the rest `test`. The rule depends only on the qids, so regeneration reproduces it. The test split is held out and reported once, at the end (M10).

## Known limitations

- 150 questions give wide confidence intervals; differences of a few points between configurations are usually noise. Report bootstrap intervals.
- About two thirds of answers are free text and need a judge model; numeric scoring covers only the 52 rows with a parsed value.
- Some golds use unstated conventions (fiscal year naming, rounding, "unadjusted" metrics). Rows such as `fb-00605` say so in the answer itself.
- `ratio` also covers day counts (DPO, cash conversion cycle) because the contract has no days unit.
- The documents are older (2015 to 2024) than the rest of the corpus, so this set tests PDF parsing and page citations more than recency.

## License

CC BY-NC 4.0 (Patronus AI). LedgerLens is a non-commercial portfolio project; this derived file keeps the attribution above and is shared under the same terms. Do not use it commercially.

## Maintenance

Regenerate with `uv run ledgerlens datasets build --only financebench`. To move to a newer upstream revision, change `REVISION` in `src/ledgerlens/eval/datasets/financebench.py`, rebuild, and review the diff, since qids and splits are stable but labels may change.
