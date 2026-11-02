import pytest

from ledgerlens.eval import calibration, judge
from ledgerlens.eval.datasets import judge_calibration as jc
from ledgerlens.eval.judge import Verdict, parse_verdict


@pytest.mark.parametrize(
    ("text", "label", "error"),
    [
        ('{"label": "CORRECT", "reason": "matches"}', "CORRECT", False),
        ('```json\n{"label": "refusal", "reason": "declined"}\n```', "REFUSAL", False),
        ("The answer is INCORRECT because the year is wrong.", "INCORRECT", True),
        ("no idea", "INCORRECT", True),
    ],
)
def test_parse_verdict(text, label, error):
    v = parse_verdict(text)
    assert v.label == label and v.parse_error is error


def _item(cid, label, split="test"):
    return jc.CalibrationItem(
        cid=cid,
        split=split,
        financebench_id="f",
        source_model="m",
        eval_mode="e",
        question="q",
        gold_answer="g",
        model_answer=cid,
        human_label=label,
    )


def test_calibrate_reports_agreement_kappa_and_confusion(monkeypatch):
    items = [
        _item(f"c{i}", lab)
        for i, lab in enumerate(["CORRECT"] * 4 + ["INCORRECT"] * 4 + ["REFUSAL"] * 4)
    ]
    monkeypatch.setattr(calibration, "load", lambda split: items)
    # The judge gets one INCORRECT item wrong (calls it CORRECT); everything else agrees.
    wrong = {"c4"}

    def fake_judge(question, gold, answer, **kwargs):
        human = next(i.human_label for i in items if i.cid == answer)
        return Verdict(label="CORRECT" if answer in wrong else human, reason="")

    monkeypatch.setattr(calibration, "judge", fake_judge)
    report = calibration.calibrate("test")
    assert report.n == 12
    assert report.accuracy.value == pytest.approx(11 / 12)
    assert report.error_rate.value == pytest.approx(1 / 12)
    # po = 11/12, pe = (5*4 + 3*4 + 4*4) / 144 = 48/144 -> kappa = (11/12 - 1/3) / (2/3) = 0.875
    assert report.kappa.value == pytest.approx(0.875)
    assert report.confusion == [[4, 0, 0], [1, 3, 0], [0, 0, 4]]
    assert report.per_label_recall["INCORRECT"] == pytest.approx(0.75)
    assert report.kappa.ci_low <= report.kappa.value <= report.kappa.ci_high


def test_build_is_balanced_and_deterministic(monkeypatch):
    rows = [
        {
            "financebench_id": f"financebench_id_{i:05d}",
            "question": "q",
            "gold_answer": "g",
            "model_answer": f"a{i}",
            "eval_mode": "m",
            "label": lab,
        }
        for i, lab in enumerate(["Correct Answer", "Incorrect Answer", "Refusal"] * 20)
    ]
    monkeypatch.setattr(jc, "_download", lambda name: rows)
    monkeypatch.setattr(jc, "SOURCES", ("only",))
    a, b = jc.build(per_label=10), jc.build(per_label=10)
    assert a == b and len(a) == 30
    assert sum(i.split == "dev" for i in a) == 12  # 40% of 10 per label


def test_judge_calls_the_judge_model(monkeypatch):
    seen = {}

    def fake_complete(model, messages, **kwargs):
        seen["model"] = model
        seen["format"] = kwargs.get("response_format")

        class R:
            content = '{"label": "CORRECT", "reason": "ok"}'

        return R()

    monkeypatch.setattr(judge, "complete", fake_complete)
    assert judge.judge("q", "g", "a").label == "CORRECT"
    assert seen["model"] == "groq-gpt-oss-120b" and seen["format"] == {"type": "json_object"}
