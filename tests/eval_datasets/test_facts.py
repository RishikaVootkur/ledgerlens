from datetime import date

from ledgerlens.eval.datasets.facts import (
    Fact,
    corpus_filings,
    facts_for,
    match_period,
    prior_year_end,
    select_fact,
)


def test_corpus_rule_latest_10k_and_two_latest_10qs(ko_submissions, ko_facts):
    filings = corpus_filings("KO", ko_submissions, ko_facts)
    assert [(f.form, f.accession) for f in filings] == [
        ("10-K", "acc-k25"),
        ("10-Q", "acc-q3"),
        ("10-Q", "acc-q2"),
    ]
    assert [f.filing_id for f in filings] == ["KO-10K-2025FY", "KO-10Q-2026Q3", "KO-10Q-2026Q2"]
    assert filings[1].period_end == date(2026, 10, 2)


def test_filing_without_facts_has_no_filing_id(ko_submissions, ko_facts):
    ko_submissions["filings"]["recent"]["accessionNumber"][1] = "acc-unknown"
    filings = corpus_filings("KO", ko_submissions, ko_facts)
    assert filings[1].fiscal_year is None and filings[1].filing_id is None


def test_facts_are_filtered_by_accession(ko_facts):
    k25 = facts_for(ko_facts, "Revenues", "USD", "acc-k25")
    assert {f.value for f in k25} == {45000, 46000, 48000}  # not the 45999 from acc-k24


def test_quarter_matching_skips_year_to_date(ko_facts):
    q3 = facts_for(ko_facts, "Revenues", "USD", "acc-q3")
    fact = match_period(q3, "quarter", date(2026, 10, 2))
    assert fact is not None and fact.value == 13000 and fact.days == 90


def test_year_and_prior_year_matching(ko_facts):
    k25 = facts_for(ko_facts, "Revenues", "USD", "acc-k25")
    end = date(2025, 12, 31)
    assert match_period(k25, "year", end).value == 48000
    assert match_period(k25, "year", prior_year_end(end)).value == 46000
    assert match_period(k25, "quarter", end) is None


def test_instant_matching_ignores_durations(ko_facts):
    q3 = facts_for(ko_facts, "Assets", "USD", "acc-q3")
    assert match_period(q3, "instant", date(2026, 10, 2)).value == 105000
    rev = facts_for(ko_facts, "Revenues", "USD", "acc-q3")
    assert match_period(rev, "instant", date(2026, 10, 2)) is None


def test_52_53_week_prior_year_lands_within_slack():
    facts = [
        Fact("Revenues", "USD", 1.0, date(2023, 10, 1), date(2024, 9, 28), "a"),
        Fact("Revenues", "USD", 2.0, date(2024, 9, 29), date(2025, 9, 27), "a"),
    ]
    assert match_period(facts, "year", prior_year_end(date(2025, 9, 27))).value == 1.0


def test_conflicting_values_for_one_period_are_dropped():
    facts = [
        Fact("Revenues", "USD", 1.0, date(2025, 1, 1), date(2025, 12, 31), "a"),
        Fact("Revenues", "USD", 2.0, date(2025, 1, 1), date(2025, 12, 31), "a"),
    ]
    assert match_period(facts, "year", date(2025, 12, 31)) is None


def test_concept_fallback_uses_first_concept_with_a_match(ko_facts):
    chain = ("SalesRevenueNet", "OperatingIncomeLoss", "Revenues")
    fact = select_fact(ko_facts, chain, "USD", "acc-k25", "year", date(2025, 12, 31))
    assert fact is not None and fact.concept == "OperatingIncomeLoss"
    # OperatingIncomeLoss has no quarterly fact in acc-q3, so the chain falls through.
    fact = select_fact(ko_facts, chain, "USD", "acc-q3", "quarter", date(2026, 10, 2))
    assert fact is not None and fact.concept == "Revenues"
    assert (
        select_fact(ko_facts, ("Missing",), "USD", "acc-q3", "quarter", date(2026, 10, 2)) is None
    )


def test_prior_year_end_handles_leap_day():
    assert prior_year_end(date(2024, 2, 29)) == date(2023, 2, 28)
