import json

import pytest
from typer.testing import CliRunner

from ledgerlens.agent.tools import EvidenceItem, EvidenceRegistry
from ledgerlens.cli import app
from ledgerlens.config import get_settings
from ledgerlens.eval import runner
from ledgerlens.llm.cache import ResponseCache
from ledgerlens.schemas import AgentAnswer, Citation, Evidence, Question, RunFile


def q(qid="x1", value=416_161e6, dataset="xbrl", answerable=True, evidence=None):
    return Question(
        qid=qid, dataset=dataset, split="dev", question="Apple net sales FY2025?",
        answerable=answerable, answer="$416,161 million" if answerable else None,
        answer_value=value if answerable else None, answer_unit="USD" if answerable else None,
        ticker="AAPL", fiscal_year=2025, question_type="single_fact",
        evidence=evidence if evidence is not None else [Evidence(filing_id="AAPL-10K-2025FY")],
    )  # fmt: skip


def registry():
    reg = EvidenceRegistry()
    reg.add(EvidenceItem("", "chunk", "AAPL-10K-2025FY:section:00001", "AAPL-10K-2025FY", "Item 7", 21, 22,
                         "Total net sales were $416,161 million in 2025.", "h"))  # fmt: skip
    reg.add(EvidenceItem("", "chunk", "KO-10K-2025FY:section:00002", "KO-10K-2025FY", "Item 7", 3, 3,
                         "Unrelated passage.", "h"))  # fmt: skip
    return reg


def answer(value=416_161e6, refused=False, cite="AAPL-10K-2025FY:section:00001"):
    cits = (
        []
        if refused
        else [Citation(chunk_id=cite, filing_id=cite.split(":")[0], section="s", quote="q")]
    )
    return AgentAnswer(answer="x" if refused else "Net sales were $416,161 million [1].", refused=refused,
                       answer_value=None if refused else value, answer_unit=None if refused else "USD",
                       citations=cits)  # fmt: skip


def test_score_correct_cited_answer():
    s = runner.score(q(), answer(), registry())
    assert s["numeric_correct"] == 1.0 and s["refusal_correct"] == 1.0
    assert s["citation_precision"] == 1.0 and s["citation_recall"] == 1.0
    assert s["recall_at_5"] == 1.0 and s["mrr"] == 1.0


def test_score_wrong_citation_and_value():
    s = runner.score(q(), answer(value=1e9, cite="KO-10K-2025FY:section:00002"), registry())
    assert s["numeric_correct"] == 0.0
    assert s["citation_precision"] == 0.0 and s["citation_recall"] == 0.0


def test_score_unanswerable():
    refused = runner.score(
        q(dataset="unanswerable", answerable=False, evidence=[]), answer(refused=True), registry()
    )
    assert refused["refusal_correct"] == 1.0
    assert refused["numeric_correct"] is None and refused["citation_precision"] is None
    hallucinated = runner.score(
        q(dataset="unanswerable", answerable=False, evidence=[]), answer(), registry()
    )
    assert hallucinated["refusal_correct"] == 0.0


def test_item_keys_cover_pages_and_doc_names():
    item = EvidenceItem(
        "", "chunk", "FB-3M:section:00001", "FB-3M", "s", 4, 5, "t", "h", doc_name="3M_2018_10K"
    )
    keys = runner.item_keys(item)
    assert {
        "3M_2018_10K#p4",
        "3M_2018_10K#p5",
        "FB-3M#p4",
        "FB-3M:section:00001",
        "3M_2018_10K",
    } <= keys


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for name in ("CACHE_DIR", "EVAL_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    get_settings.cache_clear()
    ds = tmp_path / "eval_dir" / "datasets"
    ds.mkdir(parents=True)
    rows = [q("s1"), q("s2", value=1.0), q("s3")]
    for r in rows:
        r.dataset = "smoke"
    (ds / "smoke.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows))

    def fake_agent(question, config, filters, registry):
        for item in registry_items():
            registry.add(item)
        if "boom" in question:
            raise RuntimeError("boom")
        return answer()

    def registry_items():
        return list(registry().items.values())

    monkeypatch.setattr(runner, "run_agent", fake_agent)
    yield tmp_path
    get_settings.cache_clear()


def test_run_eval_scores_summarizes_and_writes(isolated):
    cfg = runner.EvalConfig(name="t", datasets=["smoke"], concurrency=2)
    run = runner.run_eval(cfg, "dev", run_id="r1")
    assert [r.qid for r in run.results] == ["s1", "s2", "s3"]
    assert run.summary["numeric_correct"].value == pytest.approx(2 / 3)
    assert run.summary["numeric_correct"].n == 3
    assert run.summary_by_dataset["smoke"]["refusal_correct"].value == 1.0
    path = runner.write_run(run)
    assert RunFile.model_validate_json(path.read_text()).run_id == "r1"
    limited = runner.run_eval(cfg.model_copy(update={"limit": 1}), "dev")
    assert len(limited.results) == 1


def test_errors_are_recorded_not_raised(isolated, monkeypatch):
    monkeypatch.setattr(
        runner, "run_agent", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    )
    run = runner.run_eval(runner.EvalConfig(name="t", datasets=["smoke"]), "dev")
    assert all(r.error == "RuntimeError: down" for r in run.results)
    assert run.summary["error_rate"].value == 1.0


def test_record_cache_copies_touched_entries(isolated):
    s = get_settings()
    ResponseCache(s.cache_dir).put("llm", "ab" * 32, {"content": "x"})
    ResponseCache(s.cache_dir).put("embed", "cd" * 32, {"vector": [1]})
    assert runner.record_cache() >= 1
    assert (s.eval_dir / "cache" / "llm" / "ab" / f"{'ab' * 32}.json").exists()
    assert not (s.eval_dir / "cache" / "embed").exists()


def test_cli_run_compare_and_gate(isolated, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("name: t\ndatasets: [smoke]\n")
    cli = CliRunner()
    a = cli.invoke(app, ["eval", "run", "--config", str(cfg), "--run-id", "a"])
    assert a.exit_code == 0, a.output
    b = cli.invoke(app, ["eval", "run", "--config", str(cfg), "--run-id", "b"])
    assert b.exit_code == 0
    cmp = cli.invoke(app, ["eval", "compare", "a", "b"])
    assert cmp.exit_code == 0 and "numeric_correct" in cmp.output
    gate_yaml = tmp_path / "gate.yaml"
    gate_yaml.write_text("max_drop:\n  numeric_correct: 0.0\n")
    ok = cli.invoke(
        app, ["eval", "gate", "--baseline", "a", "--candidate", "b", "--thresholds", str(gate_yaml)]
    )
    assert ok.exit_code == 0, ok.output
    worse = RunFile.model_validate_json((get_settings().eval_dir / "runs" / "b.json").read_text())
    worse.results[0].scores["numeric_correct"] = 0.0
    (tmp_path / "worse.json").write_text(worse.model_dump_json())
    bad = cli.invoke(app, ["eval", "gate", "--baseline", "a", "--candidate", str(tmp_path / "worse.json"),
                           "--thresholds", str(gate_yaml)])  # fmt: skip
    assert bad.exit_code == 1 and "FAIL" in bad.output
    assert (
        json.loads((get_settings().eval_dir / "runs" / "a.json").read_text())["config"]["name"]
        == "t"
    )


def test_small_numeric_values_can_be_supported():
    reg = EvidenceRegistry()
    reg.add(
        EvidenceItem(
            "",
            "chunk",
            "c1",
            "AAPL-10K-2025FY",
            "s",
            1,
            1,
            "Diluted EPS was $6.75; gross margin 46.9%.",
            "h",
        )
    )
    for value in (6.75, 46.9):
        a = answer(value=value, cite="c1")
        assert runner.score(q(value=value), a, reg)["citation_precision"] == 1.0


def test_text_answers_use_the_support_check_and_judge_failures_are_isolated(monkeypatch, isolated):
    fb = q("fb1", dataset="financebench")
    fb.answer_value, fb.answer_unit, fb.answer = None, None, "Yes, it improved."
    text_answer = answer(cite="AAPL-10K-2025FY:section:00001").model_copy(
        update={"answer_value": None, "answer_unit": None}
    )
    reg = registry()
    assert runner.score(fb, text_answer, reg)["citation_precision"] is None
    assert runner.score(fb, text_answer, reg, lambda c, p: True)["citation_precision"] == 1.0
    assert runner.score(fb, text_answer, reg, lambda c, p: False)["citation_precision"] == 0.0

    monkeypatch.setattr(runner, "run_agent", lambda *a, **k: text_answer)
    monkeypatch.setattr(
        runner, "judge_support", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("judge down"))
    )
    monkeypatch.setattr(
        runner, "run_judge", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("judge down"))
    )
    result = runner.run_one(fb, runner.EvalConfig(name="t", datasets=["financebench"]))
    assert result.error is None
    assert result.scores["judge_correct"] is None
    assert result.scores["refusal_correct"] == 1.0


def test_xbrl_fact_from_the_right_filing_counts_as_retrieved():
    reg = EvidenceRegistry()
    reg.add(
        EvidenceItem(
            "",
            "fact",
            "xbrl:AAPL:Revenues:2024-09-29_2025-09-27:USD",
            "AAPL-10K-2025FY",
            "XBRL companyfacts",
            None,
            None,
            "Revenue 416,161,000,000 USD",
            "x",
        )
    )
    s = runner.score(q(), answer(cite="xbrl:AAPL:Revenues:2024-09-29_2025-09-27:USD"), reg)
    assert (
        s["recall_at_5"] == 1.0 and s["citation_recall"] == 1.0 and s["citation_precision"] == 1.0
    )


def test_merge_runs_combines_disjoint_parts(isolated):
    cfg = runner.EvalConfig(name="t", datasets=["smoke"])
    a = runner.run_eval(cfg.model_copy(update={"qids": ["s1"]}), "dev", run_id="a")
    b = runner.run_eval(cfg.model_copy(update={"qids": ["s2", "s3"]}), "dev", run_id="b")
    merged = runner.merge_runs([a, b], "ab")
    assert [r.qid for r in merged.results] == ["s1", "s2", "s3"]
    assert merged.summary["numeric_correct"].n == 3
    assert merged.config["merged_from"] == ["a", "b"]
    with pytest.raises(ValueError):
        runner.merge_runs([a, a], "dup")
    other = runner.run_eval(cfg.model_copy(update={"qids": ["s2"], "k": 9}), "dev", run_id="c")
    with pytest.raises(ValueError):
        runner.merge_runs([a, other], "mixed")


def test_derived_answers_are_supported_by_cited_inputs():
    reg = EvidenceRegistry()
    reg.add(
        EvidenceItem(
            "",
            "fact",
            "f24",
            "KO-10K-2025FY",
            "XBRL",
            None,
            None,
            "10,631,000,000 USD for 2024",
            "x",
        )
    )
    reg.add(
        EvidenceItem(
            "",
            "fact",
            "f25",
            "KO-10K-2025FY",
            "XBRL",
            None,
            None,
            "13,107,000,000 USD for 2025",
            "x",
        )
    )
    reg.add(
        EvidenceItem(
            "", "chunk", "c9", "KO-10K-2025FY", "s", 1, 1, "Unrelated text about bottlers.", "h"
        )
    )
    a = AgentAnswer(
        answer="Net income rose 23.29%, from $10,631 million [1] to $13,107 million [2].",
        answer_value=23.29,
        answer_unit="percent",
        citations=[
            Citation(chunk_id=c, filing_id="KO-10K-2025FY", section="s")
            for c in ("f24", "f25", "c9")
        ],
    )
    s = runner.score(q(value=23.29), a, reg)
    assert s["citation_precision"] == pytest.approx(2 / 3)
