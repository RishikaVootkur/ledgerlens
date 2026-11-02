import pytest

from ledgerlens.eval.datasets.financebench import (
    fiscal_period_of,
    parse_numeric_answer,
    to_question,
)


@pytest.mark.parametrize(
    ("answer", "question", "expected"),
    [
        ("$1577.00", "capital expenditure amount (in USD millions) for 3M?", (1.577e9, "USD", 5e3)),
        ("$8.70", "net PPNE for 3M? Answer in USD billions.", (8.7e9, "USD", 5e6)),
        ("77.78", "How much does Pfizer expect to pay in USD million?", (7.778e7, "USD", 5e3)),
        ("1.9%", "capex as a % of revenue? round to one decimal place", (1.9, "percent", 0.05)),
        ("24.26", "fixed asset turnover ratio? Round to two decimals.", (24.26, "ratio", 0.005)),
        ("-0.02", "return on assets (ROA)?", (-0.02, "ratio", 0.005)),
        ("$0.55", "dividend per share paid each quarter?", (0.55, "per_share", 0.005)),
        ("$2,018 million", "Adjusted EBITDA?", (2.018e9, "USD", 5e5)),
        ("0", "restructuring costs? If not outlined then state 0.", (0.0, "ratio", 0.5)),
    ],
)
def test_parse_numeric_answer(answer, question, expected):
    value, unit, abs_tol = parse_numeric_answer(answer, question)
    assert value == pytest.approx(expected[0])
    assert unit == expected[1]
    assert abs_tol == pytest.approx(expected[2])


@pytest.mark.parametrize(
    "answer",
    [
        "The consumer segment shrunk by 0.9% organically.",
        "Yes",
        "$400,000,000 increase and more",
        "Fiscal 2022 = FY2023: 36%",
    ],
)
def test_non_numeric_answers_are_left_as_text(answer):
    assert parse_numeric_answer(answer, "anything") is None


def test_fiscal_period_from_doc_name():
    assert fiscal_period_of("3M_2018_10K", "10-K") == "FY"
    assert fiscal_period_of("3M_2023Q2_10Q", "10-Q") == "Q2"
    assert fiscal_period_of("PEPSICO_2023Q1_EARNINGS", "EARNINGS") == "Q1"
    assert fiscal_period_of("PEPSICO_2023_8K_dated-2023-05-05", "8-K") is None


ROW = {
    "financebench_id": "financebench_id_03029",
    "company": "3M",
    "doc_name": "3M_2018_10K",
    "question_type": "metrics-generated",
    "question_reasoning": "Information extraction",
    "domain_question_num": None,
    "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
    "answer": "$1577.00",
    "justification": "Directly extracted.",
    "dataset_subset_label": "OPEN_SOURCE",
    "evidence": [
        {
            "evidence_text": "Purchases of property, plant and equipment (PP&E) (1,577)",
            "doc_name": "3M_2018_10K",
            "evidence_page_num": 59,
            "evidence_text_full_page": "...",
        }
    ],
    "gics_sector": "Industrials",
    "doc_type": "10k",
    "doc_period": 2018,
    "doc_link": "https://example.invalid/3m.pdf",
}


def test_row_maps_to_question_with_one_based_page():
    q = to_question(ROW)
    assert q.qid == "fb-03029" and q.dataset == "financebench"
    assert q.question_type == "metrics_generated"
    assert q.form == "10-K" and q.fiscal_year == 2018 and q.fiscal_period == "FY"
    assert q.ticker == "MMM" and q.doc_name == "3M_2018_10K"
    assert q.evidence[0].page == 60  # 0-based 59 -> 1-based 60
    assert q.answer_value == pytest.approx(1.577e9) and q.answer_unit == "USD"
    assert q.metadata["question_reasoning"] == "Information extraction"


def test_unit_override_for_zero_restructuring_costs():
    row = {
        **ROW,
        "financebench_id": "financebench_id_01319",
        "question": "What is the quantity of restructuring costs? If not outlined then state 0.",
        "answer": "0",
    }
    q = to_question(row)
    assert (q.answer_value, q.answer_unit) == (0.0, "USD")
