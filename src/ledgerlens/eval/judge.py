"""LLM judge for free-text answers, and its calibration against human labels.

The judge compares an answer with the gold answer and returns one of the labels
FinanceBench's human annotators used: CORRECT, INCORRECT or REFUSAL. It runs on a
different model family (GPT-OSS on Groq) from every answer model under test (Gemini,
Qwen), so it never grades its own family. Calibration measures its agreement with
FinanceBench's human labels on answers from GPT-4, Llama 2 and Claude 2.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel

from ledgerlens.llm import DEFAULT_JUDGE_MODEL, complete

Label = Literal["CORRECT", "INCORRECT", "REFUSAL"]
LABELS: tuple[Label, ...] = ("CORRECT", "INCORRECT", "REFUSAL")

# v2 (tuned on the calibration dev split only): refusals take priority over
# correctness, and empty or off-topic text counts as incorrect, matching how
# FinanceBench's annotators labeled answers.
JUDGE_PROMPT_VERSION = "v2"

JUDGE_SYSTEM = """You grade answers to questions about company financial filings.
Compare the candidate answer with the gold answer and reply with one label:
- REFUSAL: the candidate says it cannot answer, or that the information is not
  available, not provided or not disclosed (including "as an AI I don't have access").
  Use REFUSAL whenever the candidate declines, even if it adds general background, and
  even though the gold answer exists. Check this first.
- CORRECT: the candidate commits to an answer that states the gold answer's key facts
  and numbers. Rounding, units written differently (e.g. $1,577 million vs $1.58
  billion) and extra correct detail are fine. For yes/no questions the conclusion must
  match.
- INCORRECT: the candidate commits to an answer but a key number or conclusion is
  wrong, contradicts the gold answer, or is about a different company or period. Text
  that neither answers nor declines (empty, cut off, off-topic) is also INCORRECT.
Return JSON only: {"label": "CORRECT|INCORRECT|REFUSAL", "reason": "one sentence"}"""

JUDGE_USER = """Question: {question}

Gold answer: {gold}

Candidate answer: {answer}"""


class Verdict(BaseModel):
    label: Label
    reason: str
    parse_error: bool = False


def parse_verdict(text: str) -> Verdict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            label = str(data.get("label", "")).strip().upper()
            if label in LABELS:
                return Verdict(label=label, reason=str(data.get("reason", ""))[:300])  # type: ignore[arg-type]
        except json.JSONDecodeError:
            pass
    for label in LABELS:  # fall back to a bare label in prose
        if re.search(rf"\b{label}\b", text.upper()):
            return Verdict(label=label, reason=text[:300], parse_error=True)
    return Verdict(
        label="INCORRECT", reason=f"unparseable judge output: {text[:200]}", parse_error=True
    )


SUPPORT_SYSTEM = """You check citations in answers about company filings. Given a claim
from an answer and a cited passage, decide whether the passage supports the claim: the
passage must state the facts or numbers the claim relies on (rounding and unit wording
differences are fine). Return JSON only: {"supported": true|false, "reason": "one sentence"}"""


def judge_support(claim: str, passage: str, model: str = DEFAULT_JUDGE_MODEL) -> bool:
    """Whether a cited passage supports an answer's claim (semantic citation precision)."""
    resp = complete(
        model,
        [
            {"role": "system", "content": SUPPORT_SYSTEM},
            {"role": "user", "content": f"Claim: {claim[:1200]}\n\nPassage: {passage[:2500]}"},
        ],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
        purpose="citation_support",
    )
    match = re.search(r"\{.*\}", resp.content, re.DOTALL)
    if match:
        try:
            return bool(json.loads(match.group(0)).get("supported") is True)
        except json.JSONDecodeError:
            pass
    return bool(re.search(r'"?supported"?\s*:\s*true', resp.content, re.IGNORECASE))


def judge(question: str, gold: str, answer: str, model: str = DEFAULT_JUDGE_MODEL) -> Verdict:
    resp = complete(
        model,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_USER.format(
                    question=question[:2000], gold=gold[:1500], answer=answer[:2500]
                ),
            },
        ],
        temperature=0.0,
        max_tokens=400,
        response_format={"type": "json_object"},
        purpose="judge",
    )
    return parse_verdict(resp.content)
