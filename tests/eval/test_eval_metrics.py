"""Scoring metrics against hand-computed values. The arithmetic is in the comments."""

from __future__ import annotations

import math

import pytest

from ledgerlens.eval.metrics import (
    chunk_keys,
    citation_keys,
    citation_precision,
    citation_recall,
    evidence_keys,
    extract_answer_value,
    find_numbers,
    hit_at_k,
    is_refusal,
    mrr,
    ndcg_at_k,
    numeric_correct,
    numeric_match,
    numeric_match_kind,
    parse_number,
    recall_at_k,
    refusal_correct,
)
from ledgerlens.schemas import AgentAnswer, Chunk, Citation, Evidence, Question

# ---------------------------------------------------------------- retrieval

# Five retrieved items, relevant ones at ranks 2 and 4.
RANKED = ["x1", "r1", "x2", "r2", "x3"]
RELEVANT = {"r1", "r2"}


def test_recall_at_k_hand_computed() -> None:
    # top-1 = {x1}: 0 of 2 relevant. top-3 = {x1, r1, x2}: 1 of 2. top-5: 2 of 2.
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 3) == 0.5
    assert recall_at_k(RANKED, RELEVANT, 5) == 1.0
    # k past the end of the list is the whole list.
    assert recall_at_k(RANKED, RELEVANT, 50) == 1.0


def test_hit_at_k_hand_computed() -> None:
    # First relevant item is at rank 2.
    assert hit_at_k(RANKED, RELEVANT, 1) == 0.0
    assert hit_at_k(RANKED, RELEVANT, 2) == 1.0


def test_mrr_hand_computed() -> None:
    # First relevant item at rank 2 -> 1/2.
    assert mrr(RANKED, RELEVANT) == 0.5
    # MRR@1 truncates before rank 2 -> 0.
    assert mrr(RANKED, RELEVANT, k=1) == 0.0
    assert mrr(["r2"], RELEVANT) == 1.0
    assert mrr(["x1", "x2"], RELEVANT) == 0.0


def test_ndcg_hand_computed() -> None:
    # DCG@5  = 1/log2(2+1) + 1/log2(4+1) = 0.630930 + 0.430677 = 1.061606
    # IDCG@5 uses min(5, |relevant|=2) ideal ranks: 1/log2(2) + 1/log2(3) = 1 + 0.630930
    #        = 1.630930
    # nDCG@5 = 1.061606 / 1.630930 = 0.650921
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(expected)
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.650921, abs=1e-6)
    # nDCG@3 sees only r1 at rank 2. IDCG@3 is still over 2 ideal ranks.
    # 0.630930 / 1.630930 = 0.386853
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.386853, abs=1e-6)
    # Perfect ranking scores 1.
    assert ndcg_at_k(["r1", "r2", "x1"], RELEVANT, 3) == pytest.approx(1.0)
    # More relevant than k: ideal is min(k, |relevant|) = 1 rank, so a hit at rank 1 is perfect.
    assert ndcg_at_k(["r1"], RELEVANT, 1) == pytest.approx(1.0)


def test_duplicates_count_only_first_occurrence() -> None:
    # ["a", "a", "b"]: the second "a" keeps rank 2 but earns no gain.
    retrieved = ["a", "a", "b"]
    relevant = {"a", "b"}
    # top-2 covers only {a} -> 1/2.
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    # DCG@3 = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5; IDCG = 1 + 1/log2(3) = 1.630930
    # nDCG = 1.5 / 1.630930 = 0.919721, never above 1 despite the duplicate.
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(0.919721, abs=1e-6)
    assert ndcg_at_k(["a", "a"], {"a"}, 2) == pytest.approx(1.0)


def test_key_set_items_cover_several_pages() -> None:
    # One chunk spanning pages 3 and 4, both labeled.
    relevant = {"d#p3", "d#p4"}
    retrieved = [{"d", "d#p3", "d#p4"}, {"d", "d#p9"}]
    assert recall_at_k(retrieved, relevant, 1) == 1.0
    assert mrr(retrieved, relevant) == 1.0
    # Documented limitation: one item earns one gain. DCG = 1, IDCG@2 = 1.630930.
    assert ndcg_at_k(retrieved, relevant, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    # A second chunk on an already matched page earns nothing.
    assert ndcg_at_k([{"d#p3"}, {"d#p3"}], {"d#p3"}, 2) == pytest.approx(1.0)


def test_empty_relevant_returns_none() -> None:
    for fn in (recall_at_k, hit_at_k, ndcg_at_k):
        assert fn(RANKED, set(), 5) is None
    assert mrr(RANKED, set()) is None


def test_empty_retrieved_scores_zero() -> None:
    assert recall_at_k([], RELEVANT, 5) == 0.0
    assert ndcg_at_k([], RELEVANT, 5) == 0.0
    assert mrr([], RELEVANT) == 0.0


@pytest.mark.parametrize("fn", [recall_at_k, hit_at_k, ndcg_at_k])
def test_k_must_be_positive(fn: object) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        fn(RANKED, RELEVANT, 0)  # type: ignore[operator]


# ---------------------------------------------------------------- relevance keys


def _question(**kwargs: object) -> Question:
    base: dict[str, object] = {
        "qid": "q-1",
        "dataset": "financebench",
        "split": "dev",
        "question": "What was revenue?",
        "question_type": "single_fact",
    }
    base.update(kwargs)
    return Question.model_validate(base)


def _chunk(**kwargs: object) -> Chunk:
    base: dict[str, object] = {
        "chunk_id": "MMM-10K-2018FY:fixed:00007",
        "filing_id": "MMM-10K-2018FY",
        "ticker": "MMM",
        "company": "3M",
        "form": "10-K",
        "fiscal_year": 2018,
        "fiscal_period": "FY",
        "strategy": "fixed",
        "section": "Item 8",
        "text": "...",
    }
    base.update(kwargs)
    return Chunk.model_validate(base)


def test_evidence_keys_per_dataset_style() -> None:
    q = _question(
        evidence=[
            Evidence(doc_name="3M_2018_10K", page=52),  # FinanceBench
            Evidence(doc_name="3M_2018_10K", filing_id="MMM-10K-2018FY", page=53),  # doc wins
            Evidence(filing_id="AAPL-10K-2025FY", page=7),  # page, no doc name
            Evidence(filing_id="MSFT-10K-2025FY"),  # XBRL, not located
            Evidence(filing_id="NVDA-10K-2025FY", chunk_ids=["c1", "c2"]),  # XBRL, located
            Evidence(text="orphan passage"),  # no identifier: ignored
        ]
    )
    assert evidence_keys(q) == {
        "3M_2018_10K#p52",
        "3M_2018_10K#p53",
        "AAPL-10K-2025FY#p7",
        "MSFT-10K-2025FY",
        "c1",
        "c2",
    }
    assert evidence_keys(_question()) == set()


def test_chunk_keys_cover_page_range_and_both_ids() -> None:
    chunk = _chunk(doc_name="3M_2018_10K", page=52, page_end=54)
    assert chunk_keys(chunk) == {
        "MMM-10K-2018FY:fixed:00007",
        "MMM-10K-2018FY",
        "3M_2018_10K",
        *(f"MMM-10K-2018FY#p{p}" for p in (52, 53, 54)),
        *(f"3M_2018_10K#p{p}" for p in (52, 53, 54)),
    }
    # No page: only the chunk and filing-level keys.
    assert chunk_keys(_chunk()) == {"MMM-10K-2018FY:fixed:00007", "MMM-10K-2018FY"}
    # page_end missing or before page: just the one page.
    assert "MMM-10K-2018FY#p9" in chunk_keys(_chunk(page=9, page_end=3))
    assert "MMM-10K-2018FY#p3" not in chunk_keys(_chunk(page=9, page_end=3))


def test_chunk_keys_match_evidence_keys() -> None:
    q = _question(evidence=[Evidence(doc_name="3M_2018_10K", page=53)])
    hit = _chunk(doc_name="3M_2018_10K", page=52, page_end=54)
    miss = _chunk(doc_name="3M_2018_10K", page=60)
    ranked = [chunk_keys(miss), chunk_keys(hit)]
    assert mrr(ranked, evidence_keys(q)) == 0.5
    # XBRL filing-level evidence: any chunk from the filing counts.
    xq = _question(evidence=[Evidence(filing_id="MMM-10K-2018FY")])
    assert hit_at_k([chunk_keys(miss)], evidence_keys(xq), 1) == 1.0


def test_citation_keys() -> None:
    cite = Citation(chunk_id="c9", filing_id="F", section="s", page=4, page_end=5)
    assert citation_keys(cite) == {"c9", "F", "F#p4", "F#p5"}
    assert citation_keys(Citation(chunk_id="c9", filing_id="F", section="s")) == {"c9", "F"}
    chunk = _chunk(doc_name="3M_2018_10K", page=1)
    assert citation_keys(cite, chunk) == chunk_keys(chunk)


# ---------------------------------------------------------------- numbers

PARSE_CASES: list[tuple[str, float | None, float | None]] = [
    # text, parse_number (as written), extract_answer_value (base units)
    ("$1,234.5 million", 1234.5, 1_234_500_000.0),
    ("(1,234)", -1234.0, -1234.0),
    ("$(1.2) million", -1.2, -1_200_000.0),
    ("($5.0 billion)", -5.0, -5e9),
    ("-12.3%", -12.3, -12.3),
    ("−7.5 percent", -7.5, -7.5),  # Unicode minus
    ("-$300 thousand", -300.0, -300_000.0),
    ("$-300 thousand", -300.0, -300_000.0),
    ("1.2 billion", 1.2, 1.2e9),
    ("USD 3.4B", 3.4, 3.4e9),
    ("US$ 12.75 bn", 12.75, 12.75e9),
    ("1.5 trillion dollars", 1.5, 1.5e12),
    ("5k", 5.0, 5000.0),
    ("3 MM", 3.0, 3e6),
    ("0.45x", 0.45, 0.45),
    ("Revenue was 1,234.", 1234.0, 1234.0),  # trailing period
    ("1,234,567", 1_234_567.0, 1_234_567.0),
    (".5%", 0.5, 0.5),
    ("12.5 per cent", 12.5, 12.5),
    ("FY2023 Q3 margin 12 pct", 12.0, 12.0),  # FY2023 and Q3 are ids, not numbers
    ("$3.45 per share", 3.45, 3.45),
    ("In fiscal 2023, revenue was $5 billion.", 5.0, 5e9),  # marked number beats the year
    ("Net income rose 8% to $2.1 billion in 2024", 8.0, 8.0),  # first marked number wins
    ("Per the 10-K, the count was 42", 42.0, 42.0),  # 10-K is a form name
    ("2023-2024 change: 15", 15.0, 15.0),  # hyphen between years is not a minus sign
    ("in 2022 the ratio was 1.73", 1.73, 1.73),  # bare number beats a year
    ("2019", 2019.0, 2019.0),  # a lone year is still a number
    ("about 7 to 9 months", 9.0, 9.0),  # a range resolves to its end
    # Company names are not amounts (review blocker): 3M is not 3 million.
    ("3M's total revenue in FY2018 was $32.8 billion", 32.8, 32.8e9),
    ("3M's capex ratio was 0.45", 0.45, 0.45),
    ("3M had 1,577 million", 1577.0, 1.577e9),
    ("3M revenue was $3M", 3.0, 3e6),  # with a currency marker, $3M is an amount
    ("The 5th largest, at 7", 7.0, 7.0),  # a number glued to a word is skipped
    # A parenthesized year is a label, not an accounting negative.
    ("Net income (2022): 1,234", 1234.0, 1234.0),
    ("(2022)", 2022.0, 2022.0),
    # A hyphen between numbers is a range, not a minus sign.
    ("10 - 12", 12.0, 12.0),
    ("2019 - 2021", 2021.0, 2021.0),
    ("Margin fell from 30% to 25%", 25.0, 25.0),
    ("10-12%", 12.0, 12.0),
    ("from $5 to $6 billion", 6.0, 6e9),
    # A rate followed by an amount is not a range.
    ("Revenue grew 12% to $5.6 billion", 12.0, 12.0),
    ("USD3.4B", 3.4, 3.4e9),
    ("50 basis points", 50.0, 0.5),
    ("75 bps", 75.0, 0.75),
    # Documented limitation: sign words are not read.
    ("a loss of $5 million", 5.0, 5e6),
    ("no number here", None, None),
    ("", None, None),
]


@pytest.mark.parametrize(("text", "as_written", "base"), PARSE_CASES)
def test_parse_number_table(text: str, as_written: float | None, base: float | None) -> None:
    assert parse_number(text) == as_written
    if base is None:
        assert extract_answer_value(text) is None
    else:
        assert extract_answer_value(text) == pytest.approx(base)


def test_find_numbers_records_markers_and_precision() -> None:
    [n] = find_numbers("$1.25 billion")
    assert (n.value, n.scale, n.decimals, n.currency, n.marked) == (1.25, 1e9, 2, True, True)
    assert n.base_value == pytest.approx(1.25e9)
    [p] = find_numbers("12.5%")
    assert p.percent and not p.currency and p.decimals == 1
    [y] = find_numbers("2023")
    assert y.looks_like_year
    start, end = find_numbers("10 - 12")
    assert (start.value, start.range_end, end.value, end.range_end) == (10, False, 12, True)
    # A minus that does not follow a number is still a sign.
    [neg] = find_numbers("change of - 12")
    assert neg.value == -12


def test_extract_answer_value_uses_unit_to_choose() -> None:
    text = "Revenue rose from $5.0 billion to $6.0 billion, a 20% increase."
    # "from X to Y": the end of the range is the headline.
    assert extract_answer_value(text) == pytest.approx(6e9)
    assert extract_answer_value(text, "percent") == pytest.approx(20.0)
    assert extract_answer_value("Up 3% to 1.2 million", "USD") == pytest.approx(1.2e6)
    # For USD a currency-marked number beats a scale-only one.
    assert extract_answer_value("1.5 million shares for $40 billion", "USD") == pytest.approx(40e9)
    assert extract_answer_value("Revenue grew 12% to $5.6 billion", "USD") == pytest.approx(5.6e9)
    assert extract_answer_value("$5 billion", "text") is None


def test_numeric_match_relative_tolerance() -> None:
    # 1% of 100 is 1: 101 is inside, 101.5 is outside.
    assert numeric_match(101.0, 100.0, 0.01, unit="USD")
    assert numeric_match(99.0, 100.0, 0.01, unit="USD")
    assert not numeric_match(101.5, 100.0, 0.01, unit="USD")


def test_numeric_match_never_forgives_scale_or_sign() -> None:
    assert not numeric_match(1.2e6, 1.2e9, 0.01, unit="USD")
    assert not numeric_match(1.2e12, 1.2e9, 0.01, unit="USD")
    assert not numeric_match(-100.0, 100.0, 0.01, unit="USD")
    # The x100 allowance is for percents only.
    assert not numeric_match(0.125, 12.5, 0.01, unit="ratio")


def test_numeric_match_percent_as_fraction() -> None:
    assert numeric_match(0.125, 12.5, 0.01, unit="percent")
    assert numeric_match(12.5, 12.5, 0.01, unit="percent")
    assert not numeric_match(0.135, 12.5, 0.01, unit="percent")
    # A fraction reading needs no percent marker on the prediction ...
    assert not numeric_match(0.125, 12.5, 0.01, unit="percent", pred_percent_marked=True)
    # ... and |pred| <= 1.5: 1.25 can be a fraction (125%), 2.5 cannot.
    assert numeric_match(1.25, 125.0, 0.01, unit="percent")
    assert not numeric_match(2.5, 250.0, 0.01, unit="percent")
    # Never divided by 100: these are 100x errors (review blocker).
    assert not numeric_match(100, 1.0, 0.01, unit="percent")
    assert not numeric_match(50, 0.5, 0.01, unit="percent")
    assert not numeric_match(1250, 12.5, 0.01, unit="percent")
    assert not numeric_match(12.5, 0.125, 0.01, unit="percent")


def test_numeric_match_gold_rounding() -> None:
    # Gold text "$1.2 billion" shows tenths of a billion: half a step is 0.05e9.
    # 1.24e9 is 3.3% off (outside 1%) but rounds to $1.2 billion.
    assert numeric_match(1.24e9, 1.2e9, 0.01, unit="USD", gold_text="$1.2 billion")
    assert not numeric_match(1.24e9, 1.2e9, 0.01, unit="USD")
    # 1.26e9 rounds to 1.3 billion.
    assert not numeric_match(1.26e9, 1.2e9, 0.01, unit="USD", gold_text="$1.2 billion")
    # "12.5%" shows tenths of a point (half step 0.05). With rel_tol 0.1% the plain
    # tolerance is 0.0125 and the cap is 5 * 0.0125 = 0.0625, so the allowance is 0.05.
    assert numeric_match(12.54, 12.5, 0.001, unit="percent", gold_text="12.5%")
    assert not numeric_match(12.54, 12.5, 0.001, unit="percent")
    # With rel_tol 0 the cap is 0: rounding grants nothing.
    assert not numeric_match(12.54, 12.5, 0.0, unit="percent", gold_text="12.5%")
    # Round golds: "$1 billion" has half step 0.5e9 but the cap is 5 * 1% * 1e9 = 0.05e9.
    assert not numeric_match(1.49e9, 1e9, 0.01, unit="USD", gold_text="$1 billion")
    assert numeric_match(1.04e9, 1e9, 0.01, unit="USD", gold_text="$1 billion")
    # "5%": half step 0.5, cap 5 * 1% * 5 = 0.25.
    assert not numeric_match(5.4, 5, 0.01, unit="percent", gold_text="5%")
    assert numeric_match(5.2, 5, 0.01, unit="percent", gold_text="5%")
    # Gold text that disagrees with the gold value grants nothing.
    assert not numeric_match(101.5, 100.0, 0.01, unit="USD", gold_text="about 7")
    assert not numeric_match(101.5, 100.0, 0.01, unit="USD", gold_text="no number")


def test_numeric_match_absolute_tolerance_and_bad_predictions() -> None:
    # Gold 0: relative tolerance is 0, so only abs_tol helps.
    assert not numeric_match(0.004, 0.0, 0.01, unit="ratio")
    assert numeric_match(0.004, 0.0, 0.01, unit="ratio", abs_tol=0.005)
    assert numeric_match(0.0, 0.0, 0.01, unit="ratio")
    assert not numeric_match(None, 1.0, 0.01, unit="ratio")
    assert not numeric_match(math.nan, 1.0, 0.01, unit="ratio")
    assert not numeric_match(math.inf, 1.0, 0.01, unit="ratio")


def test_numeric_match_kind_reports_the_allowance_used() -> None:
    assert numeric_match_kind(100.5, 100.0, 0.01, unit="USD") == "tolerance"
    assert numeric_match_kind(0.125, 12.5, 0.01, unit="percent") == "fraction_as_percent"
    kind = numeric_match_kind(1.24e9, 1.2e9, 0.01, unit="USD", gold_text="$1.2 billion")
    assert kind == "gold_rounding"
    assert numeric_match_kind(2.0, 1.0, 0.01, unit="USD", gold_text="$1") is None


def test_numeric_correct_reads_percent_marker_from_text() -> None:
    q = _question(answer="12.5%", answer_value=12.5, answer_unit="percent")
    assert numeric_correct(AgentAnswer(answer="The margin was 0.125."), q) == 1.0
    # "0.125%" is a percent already, so it is not rescaled to 12.5.
    assert numeric_correct(AgentAnswer(answer="The margin was 0.125%."), q) == 0.0
    assert numeric_correct(AgentAnswer(answer="Margin fell from 30% to 12.5%."), q) == 1.0


def test_numeric_correct_from_question() -> None:
    q = _question(answer="$1.2 billion", answer_value=1.2e9, answer_unit="USD")
    assert numeric_correct(AgentAnswer(answer="It was $1.24 billion."), q) == 1.0
    assert numeric_correct(AgentAnswer(answer="It was $1.2 million."), q) == 0.0
    # The agent's own parsed value wins over the text.
    assert numeric_correct(AgentAnswer(answer="see table", answer_value=1.2e9), q) == 1.0
    assert numeric_correct(AgentAnswer(answer="no idea"), q) == 0.0
    # Text and unanswerable questions are not numerically scored.
    text_q = _question(answer="Yes", answer_value=None, answer_unit="text")
    assert numeric_correct(AgentAnswer(answer="Yes"), text_q) is None
    assert numeric_correct(AgentAnswer(answer="x"), _question(answerable=False)) is None


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    "text",
    [
        "This cannot be determined from the filings.",
        "The segment margin is not disclosed.",
        "I don't have enough information to answer.",
        "I don’t have enough information to answer.",
        "The filings do not contain a breakdown by region.",
        "The company does not disclose this figure.",
        "That figure is not available in the provided documents.",
        "I was unable to find the number in the 10-K.",
        "There is insufficient information to compute the ratio.",
        "No information is available on this.",
    ],
)
def test_is_refusal_positive(text: str) -> None:
    assert is_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Revenue was $5.0 billion in fiscal 2024 [1].",
        "The filing contains a detailed segment breakdown.",
        "Operating margin determined by segment was 12%.",
    ],
)
def test_is_refusal_negative(text: str) -> None:
    assert not is_refusal(text)


def test_refusal_correct_truth_table() -> None:
    answered = AgentAnswer(answer="Revenue was $5 billion.")
    flagged = AgentAnswer(answer="Revenue was $5 billion.", refused=True)
    prose = AgentAnswer(answer="This is not disclosed in the filings.")
    # Answerable: answering is right, refusing (flag or prose) is wrong.
    assert refusal_correct(answered, answerable=True)
    assert not refusal_correct(flagged, answerable=True)
    assert not refusal_correct(prose, answerable=True)
    # Unanswerable: refusing is right, answering is wrong.
    assert not refusal_correct(answered, answerable=False)
    assert refusal_correct(flagged, answerable=False)
    assert refusal_correct(prose, answerable=False)


# ---------------------------------------------------------------- citations


def test_citation_precision() -> None:
    # 3 of 4 citations supported -> 0.75.
    assert citation_precision([True, True, False, True]) == 0.75
    assert citation_precision([False]) == 0.0
    assert citation_precision([]) is None


def test_citation_recall() -> None:
    # Evidence {p1, p2, p3}; cited {p1, p3, p9} covers 2 of 3 -> 0.6667.
    assert citation_recall({"p1", "p3", "p9"}, {"p1", "p2", "p3"}) == pytest.approx(2 / 3)
    assert citation_recall(set(), {"p1"}) == 0.0
    assert citation_recall({"p1"}, set()) is None
