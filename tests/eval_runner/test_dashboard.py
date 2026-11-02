import json

import pytest
import yaml

from ledgerlens.config import get_settings
from ledgerlens.eval import dashboard
from ledgerlens.schemas import MetricValue, QuestionResult, RunFile


def run(run_id, correct, latency=100.0, category=None):
    results = [
        QuestionResult(
            qid=f"q{i}",
            dataset="xbrl",
            question="?",
            gold="1",
            prediction=None,
            scores={"numeric_correct": float(c)},
            failure_category=category if not c else None,
            failure_note="why" if not c else None,
        )
        for i, c in enumerate(correct)
    ]
    mean = sum(correct) / len(correct)
    return RunFile(
        run_id=run_id,
        created_at="2026-11-01T00:00:00+00:00",
        config_name=run_id,
        config={},
        split="dev",
        datasets=["xbrl"],
        results=results,
        summary={
            "numeric_correct": MetricValue(value=mean, ci_low=0, ci_high=1, n=len(correct)),
            "model_latency_p50_ms": MetricValue(value=latency, n=len(correct)),
        },
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_DIR", str(tmp_path / "eval"))
    get_settings.cache_clear()
    runs = tmp_path / "eval" / "runs"
    runs.mkdir(parents=True)
    for r in (
        run("base", [1, 0, 0, 1, 0, 1], 100),
        run("better", [1, 1, 1, 1, 0, 1], 80, "wrong_period"),
    ):
        (runs / f"{r.run_id}.json").write_text(r.model_dump_json())
    spec = {
        "headline": {"run_id": "better", "title": "Default agent"},
        "tables": [{"id": "t", "title": "T", "description": "d", "baseline": "Base",
                    "columns": ["numeric_correct", "model_latency_p50_ms"],
                    "rows": [{"name": "Base", "run_id": "base"}, {"name": "Better", "run_id": "better"}]}],
        "failures_from": ["better"],
    }  # fmt: skip
    path = tmp_path / "experiments.yaml"
    path.write_text(yaml.safe_dump(spec))
    yield path
    get_settings.cache_clear()


def test_dashboard_tables_have_paired_deltas(env):
    out, run_ids = dashboard.write_dashboard(env)
    data = json.loads(out.read_text())
    assert sorted(run_ids) == ["base", "better"]
    rows = {r["name"]: r for r in data["tables"][0]["rows"]}
    better = rows["Better"]["metrics"]
    assert better["numeric_correct"]["delta"] == pytest.approx(2 / 6)
    assert 0 < better["numeric_correct"]["p_value"] <= 1  # McNemar on 2 discordant pairs = 0.5
    assert better["model_latency_p50_ms"]["delta"] == pytest.approx(-20)
    assert "delta" not in rows["Base"]["metrics"]["numeric_correct"]
    assert data["headline"]["metrics"]["numeric_correct"]["value"] == pytest.approx(5 / 6)
    assert data["failure_counts"] == {"wrong_period": 1}
    assert data["failures"][0]["explanation"] == "why"


def test_unknown_baseline_is_an_error(env):
    spec = yaml.safe_load(env.read_text())
    spec["tables"][0]["baseline"] = "Nope"
    env.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        dashboard.write_dashboard(env)
