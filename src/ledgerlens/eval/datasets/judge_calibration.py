"""Judge calibration set: model answers to FinanceBench questions with the human labels
FinanceBench's annotators assigned (results/ in github.com/patronus-ai/financebench,
CC BY-NC 4.0). Balanced across the three labels and spread over source models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

from ledgerlens.config import get_settings
from ledgerlens.eval.judge import Label

RESULTS_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/results/{name}.jsonl"
SOURCES = (
    "gpt-4_oracle",
    "gpt-4_singleStore",
    "gpt-4_sharedStore",
    "gpt-4_closedBook",
    "gpt-4-1106-preview_inContext",
    "gpt-4-1106-preview_oracle",
    "claude-2_inContext",
    "llama2_singleStore",
    "llama2_sharedStore",
)
LABEL_MAP: dict[str, Label] = {
    "Correct Answer": "CORRECT",
    "Incorrect Answer": "INCORRECT",
    "Refusal": "REFUSAL",
}
PER_LABEL = 80
DEV_FRACTION = 0.4


class CalibrationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cid: str
    split: str
    financebench_id: str
    source_model: str
    eval_mode: str
    question: str
    gold_answer: str
    model_answer: str
    human_label: Label


def _download(name: str) -> list[dict[str, object]]:
    cache = get_settings().data_dir / "raw" / "financebench_results" / f"{name}.jsonl"
    if not cache.exists():
        resp = httpx.get(RESULTS_URL.format(name=name), timeout=60)
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text)
    return [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build(per_label: int = PER_LABEL) -> list[CalibrationItem]:
    """Deterministic: within each label, items are ordered by a hash of their id and the
    first `per_label` are kept. The split is by FinanceBench question: about 40% of
    questions (by a hash of the question id) are dev, so no question has answers in both
    dev and test."""
    pool: dict[Label, list[CalibrationItem]] = {label: [] for label in LABEL_MAP.values()}
    for name in SOURCES:
        for row in _download(name):
            label = LABEL_MAP.get(str(row.get("label")))
            answer = str(row.get("model_answer") or "").strip()
            if label is None or not answer:
                continue
            fid = str(row["financebench_id"])
            cid = f"cal-{name}-{fid.rsplit('_', 1)[-1]}"
            pool[label].append(
                CalibrationItem(
                    cid=cid,
                    split="",
                    financebench_id=fid,
                    source_model=name,
                    eval_mode=str(row.get("eval_mode", "")),
                    question=str(row["question"]),
                    gold_answer=str(row["gold_answer"]),
                    model_answer=answer,
                    human_label=label,
                )
            )
    chosen: list[CalibrationItem] = []
    for items in pool.values():
        items.sort(key=lambda i: _hash(i.cid))
        chosen.extend(items[:per_label])
    questions = sorted({i.financebench_id for i in chosen}, key=lambda q: _hash("split:" + q))
    dev_questions = set(questions[: round(len(questions) * DEV_FRACTION)])
    for item in chosen:
        item.split = "dev" if item.financebench_id in dev_questions else "test"
    return sorted(chosen, key=lambda i: i.cid)


def write(items: list[CalibrationItem], path: Path | None = None) -> Path:
    path = path or get_settings().eval_dir / "datasets" / "judge_calibration.jsonl"
    path.write_text("".join(i.model_dump_json() + "\n" for i in items))
    return path


def load(split: str | None = None, path: Path | None = None) -> list[CalibrationItem]:
    path = path or get_settings().eval_dir / "datasets" / "judge_calibration.jsonl"
    items = [CalibrationItem.model_validate_json(x) for x in path.read_text().splitlines() if x]
    return [i for i in items if split is None or i.split == split]
