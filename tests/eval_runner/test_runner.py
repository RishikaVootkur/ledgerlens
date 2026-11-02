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
