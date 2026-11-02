# Datasheet: XBRL golden set (`xbrl.jsonl`)

## Motivation

A large numeric eval with exact ground truth and no human labeling. Every gold value is a fact the company itself tagged in XBRL in a filing that LedgerLens ingests, so a correct answer is always findable in the corpus and the grader can score it numerically. It covers lookups, year-over-year changes and two-fact arithmetic, the operations analysts ask for most.

## Composition

350 questions over all 30 universe companies (9 each for 27 companies; 39 AAPL, 35 KO and 33 JPM, since the slice companies are not capped). 180 questions come from 10-Ks and 170 from 10-Qs.

| question_type | dev | test | total |
|---|---:|---:|---:|
| multi_step | 43 | 65 | 108 |
| single_fact | 66 | 100 | 166 |
| yoy_change | 30 | 46 | 76 |
| **total** | 139 | 211 | 350 |

- `single_fact`: revenue, net income, operating income, total assets, cash, diluted EPS, R&D expense, and for financials net interest income and noninterest revenue. Segment revenue is skipped because companyfacts has no dimensional data.
- `yoy_change`: revenue, net income, operating income and total assets, current fiscal year against the prior year, both values from the same 10-K. Each has an absolute variant (USD) and a percent variant (skipped when the prior value is not positive).
- `multi_step`: net, gross and operating margin, current ratio and total liabilities to stockholders' equity (from the 10-K and the latest 10-Q), and the percent change in revenue and net income between the two corpus 10-Q quarters.

Units: 173 USD, 117 percent, 33 ratio, 27 per share. `answer_value` is in base units (USD, never millions); percents are stored as 12.5 for 12.5%.

## Collection

Sources are the SEC JSON APIs, cached raw under `data/raw/sec_api/` (same path layout as the URL) so reruns make no requests; requests carry `SEC_USER_AGENT` and stay at or below 4 per second.

Corpus rule, shared with ingestion: per company, the latest 10-K and the two most recent 10-Qs by filing date from `submissions/CIK##########.json` `filings.recent`. `filing_id` is `make_filing_id(ticker, form, fy, fp)`, with `fy`/`fp` taken from the companyfacts facts whose `accn` is the filing's accession number. The cache snapshot used for the committed file was fetched on 2026-11-01.

## Labeling

Fact selection, in `src/ledgerlens/eval/datasets/facts.py`:

- Only us-gaap facts whose `accn` equals a corpus filing's accession. `fy`, `fp` and `frame` describe the filing, so the fact's own `start`/`end` decide the period.
- Annual flows need a 350 to 380 day duration ending at the 10-K period end. Quarterly flows need 75 to 125 days ending at the 10-Q period end, which excludes the six- and nine-month year-to-date values that 10-Qs also report and allows 12- and 16-week retail quarters. Balance sheet items are instants on the period end. Prior-year values must end within 7 days of the same date a year earlier (52/53-week years).
- If several facts match one period with different values, the question is dropped.
- Concept fallbacks, first match wins: revenue `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet`, `RevenueFromContractWithCustomerIncludingAssessedTax`; for Financials `RevenuesNetOfInterestExpense`, then `Revenues` (asked as "total net revenue"). Net income `NetIncomeLoss`, then `NetIncomeLossAvailableToCommonStockholdersBasic` (asked as "net income attributable to common stockholders"). Cash `CashAndCashEquivalentsAtCarryingValue`, then `CashAndDueFromBanks` (asked as "cash and due from banks"). R&D `ResearchAndDevelopmentExpense`, then the variant excluding acquired in-process R&D. Changes over time reuse the concept found for the later period.
- XOM and CVX tag "Total revenues and other income" as `Revenues`, so their questions use that label.

Questions name the company, the metric, the period and the unit. Derived questions state the formula, e.g. "net profit margin (net income divided by total revenue, as a percentage)".

- Periods: fiscal year with its end date, or the quarter's length taken from the fact's own duration with its end date and fiscal quarter. 13-week and calendar quarters are "the three months ended", Costco's and PepsiCo's 12-week quarters are "the 12 weeks ended" (6 questions), and 16-week quarters would be "the 16 weeks ended", matching the column headings in those filings.
- Parent-only amounts: `NetIncomeLoss` is net income attributable to the parent. When the consolidated `ProfitLoss` in the same filing and period differs from it by more than 1%, or the noncontrolling-interest amount is more than 1% of it, the question says "net income attributable to {company}" (24 questions, including margins and changes that use it). `StockholdersEquity` excludes noncontrolling interests, so every liabilities-to-equity question says "total stockholders' equity attributable to {company} (excluding noncontrolling interests)" (17 questions).

Tolerances: `rel_tolerance` 0.01 for reported facts and 0.02 for derived values; percents also get `abs_tolerance` 0.05 points and ratios 0.005, so values near zero are not graded on noise.

`metadata` stores `metric`, `formula` (in terms of `input1`, `input2`) and for each input its concept, value, unit, period start and end, filing_id and accession. `evidence` lists `Evidence(filing_id, xbrl_concept)` for each distinct input.

Per-company caps (4 single_fact, 2 yoy_change, 3 multi_step) keep eval cost low on free LLM tiers. The kept questions are the lowest SHA-256 of `"cap:" + qid`. The salt matters: without it, capping and splitting would use the same ordering and capped companies would land mostly in dev.

## Splits

Within each `question_type`, rows are ordered by the SHA-256 hex digest of the `qid`; the first `round(0.4 * n)` are `dev`, the rest `test`. Qids are semantic (`xbrl-aapl-10k2025fy-revenue`), so the split of a question never depends on generation order. The test split is held out and reported once, at the end.

Splits are by question, so dev and test can read the same reported fact: 59 of the 211 test questions use at least one fact that some dev question also uses (for example a revenue lookup in dev and a net margin over the same revenue in test). That is harmless for a retrieval system with nothing trained on dev. If dev is ever used to pick few-shot examples or tune prompts on specific facts, split by `(ticker, filing)` instead.

## Known limitations

- Companies skipped in part: XOM's 2026 Q2 10-Q and both NEE 10-Qs have no facts in companyfacts at fetch time, so they yield no questions and their fiscal labels cannot be derived from XBRL. NEE has 10-K questions only and no revenue question (no revenue concept in the fallback chain).
- Missing concepts drop questions: banks have no operating income or gross margin, several companies (JNJ, UNH, CAT, AMT) lack some balance sheet or income concepts in these filings.
- Net income and equity are parent-only amounts. The wording says so where it matters (see Labeling), but a model that reads the consolidated "Net income" or "Total equity" line will still be graded wrong on those rows, e.g. NEE FY2025 (6,835M attributable vs 5,332M consolidated) and the AMT liabilities-to-equity ratio (equity 3,652.5M vs total equity 10,355.3M). The attributable check needs `ProfitLoss` or the NCI amount in companyfacts; a filer that tags neither gets plain "net income".
- The two corpus 10-Qs are not always adjacent quarters (MSFT fiscal 2026 Q3 and fiscal 2027 Q1, NKE likewise, because Q4 is in the 10-K). The quarter change questions name both quarters explicitly.
- The XBRL value is the ground truth even when the text of the filing rounds it differently (e.g. "$416.2 billion"); tolerances absorb normal rounding.
- If the agent's `get_company_facts` tool exposes companyfacts directly, this set measures tool use more than retrieval; compare runs with and without that tool.

## License

Derived from SEC EDGAR data, which is public and free to use. The questions and code are MIT like the rest of the repo.

## Maintenance

`uv run ledgerlens datasets build --only xbrl` regenerates this file and the smoke set from the cache. Deleting `data/raw/sec_api/` refetches newer filings, which changes the corpus and the questions; do that only together with re-ingestion, and record the new snapshot date here.
