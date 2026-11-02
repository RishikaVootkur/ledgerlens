import pytest

from ledgerlens.eval import analysis
from ledgerlens.schemas import AgentAnswer, Chunk, Citation, QuestionResult


def chunk(cid, text, table=False):
    return Chunk(
        chunk_id=cid,
        filing_id="AAPL-10K-2025FY",
        ticker="AAPL",
        company="Apple",
        form="10-K",
        fiscal_year=2025,
        fiscal_period="FY",
        strategy="section",
        section="Item 8",
        text=text,
        has_table=table,
    )


class Store:
    def __init__(self, chunks):
        self.chunks = {c.chunk_id: c for c in chunks}

    def get_many(self, ids):
        return {i: self.chunks[i] for i in ids if i in self.chunks}


@pytest.fixture
def store(monkeypatch):
    chunks = [
        chunk(
            "t1", "| | 2025 | 2024 |\n|---|---|---|\n| Net sales | 416,161 | 391,035 |", table=True
        ),
        chunk("p1", "Research and development expense was $34,550 million."),
    ]
    monkeypatch.setattr(analysis, "get_store", lambda: Store(chunks))


def result(
    value,
    gold=416_161e6,
    qtype="single_fact",
    refused=False,
    answerable=True,
    retrieved=("t1",),
    scores=None,
    citations=(),
):
    pred = AgentAnswer(
        answer="x",
        refused=refused,
        answer_value=value,
        retrieved_chunk_ids=list(retrieved),
        citations=list(citations),
    )
    base = {"numeric_correct": 0.0, "refusal_correct": 1.0}
    return QuestionResult(
        qid="q",
        dataset="xbrl",
        question="?",
        question_type=qtype,
        answerable=answerable,
        gold="g",
        gold_value=gold,
        gold_unit="USD",
        prediction=pred,
        scores=scores or base,
    )


@pytest.mark.parametrize(
    ("r", "category"),
    [
        (result(391_035e6), "wrong_period"),  # adjacent year column of the same table
        (result(None, refused=True, scores={"refusal_correct": 0.0}), "bad_refusal"),
        (result(1.0, answerable=False, gold=None, scores={"refusal_correct": 0.0}), "bad_refusal"),
        (result(123e9, retrieved=("p1",)), "retrieval_miss"),
        (result(5.0, qtype="yoy_change"), "arithmetic"),
        (result(416_161e6, scores={"numeric_correct": 1.0, "refusal_correct": 1.0}), None),
    ],
)
def test_classify(store, r, category):
    if category is None:
        assert not analysis.is_failure(r)
    else:
        assert analysis.is_failure(r)
        assert analysis.classify(r)[0] == category


def test_cited_fact_for_another_period_is_wrong_period(store):
    cit = Citation(
        chunk_id="xbrl:AAPL:Revenues:2023-10-01_2024-09-28:USD",
        filing_id="AAPL-10K-2024FY",
        section="XBRL",
        quote="391,035,000,000 USD for 2023-10-01 to 2024-09-28",
    )
    r = result(391_035e6, retrieved=(), citations=[cit])
    category, note = analysis.classify(r)
    assert category == "wrong_period" and "2023-10-01_2024-09-28" in note


def test_low_citation_precision_is_hallucinated_citation(store):
    r = result(
        416_161e6,
        scores={"numeric_correct": 0.0, "refusal_correct": 1.0, "citation_precision": 0.0},
    )
    assert analysis.classify(r)[0] == "hallucinated_citation"
