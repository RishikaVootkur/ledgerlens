import pytest

from ledgerlens.eval.datasets.xbrl import (
    CAPS,
    cap_per_type,
    company_questions,
    pct_change,
    possessive,
    quarter_span,
    ratio_value,
)
from ledgerlens.universe import BY_TICKER


@pytest.fixture
def ko_questions(ko_submissions, ko_facts):
    return {q.qid: q for q in company_questions(BY_TICKER["KO"], ko_submissions, ko_facts)}


def test_single_facts_use_matching_period_and_filing(ko_questions):
    rev = ko_questions["xbrl-ko-10k2025fy-revenue"]
    assert rev.answer_value == 48000 and rev.answer_unit == "USD" and rev.rel_tolerance == 0.01
    assert rev.evidence[0].filing_id == "KO-10K-2025FY"
    assert rev.evidence[0].xbrl_concept == "us-gaap:Revenues"
    assert rev.fiscal_year == 2025 and rev.fiscal_period == "FY" and rev.form == "10-K"
    q3 = ko_questions["xbrl-ko-10q2026q3-revenue"]
    assert q3.answer_value == 13000  # three-month value, not the 37,900 year to date
    assert "three months ended October 2, 2026" in q3.question
    assert ko_questions["xbrl-ko-10q2026q3-total_assets"].answer_value == 105000
    eps = ko_questions["xbrl-ko-10k2025fy-eps_diluted"]
    assert eps.answer_unit == "per_share" and eps.answer_value == 2.3


def test_yoy_change_absolute_and_percent(ko_questions):
    absq = ko_questions["xbrl-ko-10k2025fy-revenue-yoy-abs"]
    assert absq.answer_value == 2000 and absq.answer_unit == "USD"
    pct = ko_questions["xbrl-ko-10k2025fy-revenue-yoy-pct"]
    assert pct.answer_value == pytest.approx(4.3478, abs=1e-4)
    assert pct.answer_unit == "percent" and pct.rel_tolerance == 0.02
    assert pct.metadata["input1_value"] == 48000 and pct.metadata["input2_value"] == 46000
    assert len(pct.evidence) == 1  # both inputs are the same concept in the same 10-K
    assert ko_questions["xbrl-ko-10k2025fy-total_assets-yoy-abs"].answer_value == 10000


def test_multi_step_ratios(ko_questions):
    nm = ko_questions["xbrl-ko-10k2025fy-net_margin"]
    assert nm.answer_value == pytest.approx(20.8333, abs=1e-4)
    assert nm.metadata["formula"] == "input1 / input2 * 100"
    assert {e.xbrl_concept for e in nm.evidence} == {"us-gaap:NetIncomeLoss", "us-gaap:Revenues"}
    cr = ko_questions["xbrl-ko-10k2025fy-current_ratio"]
    assert cr.answer_value == 1.5 and cr.answer_unit == "ratio"
    assert ko_questions["xbrl-ko-10k2025fy-operating_margin"].answer_value == 25.0


def test_quarter_over_quarter_change_across_the_two_10qs(ko_questions):
    q = ko_questions["xbrl-ko-10q2026q3-revenue-qoq-pct"]
    assert q.answer_value == 4.0
    assert {e.filing_id for e in q.evidence} == {"KO-10Q-2026Q3", "KO-10Q-2026Q2"}
    assert ko_questions["xbrl-ko-10q2026q3-net_income-qoq-pct"].answer_value == pytest.approx(
        7.1429, abs=1e-4
    )


def test_missing_concepts_produce_no_question(ko_questions):
    assert "xbrl-ko-10k2025fy-rd_expense" not in ko_questions
    assert "xbrl-ko-10k2025fy-gross_margin" not in ko_questions


def test_derived_math():
    assert ratio_value(1, 4, "percent") == 25.0
    assert ratio_value(3, 2, "ratio") == 1.5
    assert ratio_value(2, 3, "ratio") == 0.6667
    assert pct_change(110, 100) == pytest.approx(10.0)
    assert pct_change(90, 100) == pytest.approx(-10.0)
    assert pct_change(-50, -100) == pytest.approx(50.0)  # loss shrinking is a rise


def test_caps_are_per_type(ko_questions):
    capped = cap_per_type(list(ko_questions.values()), CAPS)
    counts: dict[str, int] = {}
    for q in capped:
        counts[q.question_type] = counts.get(q.question_type, 0) + 1
    assert all(counts[t] <= CAPS[t] for t in counts)


def test_possessive():
    assert possessive("Apple") == "Apple's"
    assert possessive("American Express") == "American Express'"


def test_net_income_says_attributable_when_consolidated_amount_differs(ko_questions):
    # ProfitLoss 10,500 vs NetIncomeLoss 10,000 in the 10-K: beyond the 1% tolerance.
    assert "net income attributable to Coca-Cola for fiscal year 2025" in (
        ko_questions["xbrl-ko-10k2025fy-net_income"].question
    )
    assert "net income attributable to Coca-Cola" in (
        ko_questions["xbrl-ko-10k2025fy-net_margin"].question
    )
    assert "net income attributable to Coca-Cola" in (
        ko_questions["xbrl-ko-10k2025fy-net_income-yoy-pct"].question
    )
    # Q3: ProfitLoss 3,001 vs 3,000 is within tolerance, so plain wording.
    q3 = ko_questions["xbrl-ko-10q2026q3-net_income"].question
    assert "Coca-Cola's net income for the three months" in q3
    # Q2 has a material NCI amount, so the change across Q2 and Q3 names the parent.
    assert "net income attributable to Coca-Cola" in (
        ko_questions["xbrl-ko-10q2026q3-net_income-qoq-pct"].question
    )
    assert ko_questions["xbrl-ko-10k2025fy-net_income"].answer_value == 10000


def test_equity_ratios_name_parent_equity(ko_questions):
    q = ko_questions["xbrl-ko-10k2025fy-liabilities_to_equity"]
    assert q.answer_value == 3.0
    assert (
        "total stockholders' equity attributable to Coca-Cola (excluding noncontrolling "
        "interests)" in q.question
    )


@pytest.mark.parametrize(
    ("days", "span"),
    [
        (83, "12 weeks"),
        (84, "12 weeks"),
        (90, "three months"),
        (91, "three months"),
        (111, "16 weeks"),
        (None, "three months"),
    ],
)
def test_quarter_span_follows_fact_duration(days, span):
    assert quarter_span(days) == span
