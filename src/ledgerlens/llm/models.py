"""Model registry: how to reach each model and what it would cost at list price.

Prices are public list prices in USD per million tokens, checked 2026-11-01. Calls on
free tiers are still priced at list so runs can be compared on cost ("list-price
equivalent"). Sources are recorded in docs/decisions.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Provider = Literal["gemini", "groq", "ollama"]

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: Provider
    model: str
    family: str
    input_per_mtok: float
    output_per_mtok: float
    requests_per_minute: int
    tokens_per_minute: int | None
    label: str
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def litellm_model(self) -> str:
        match self.provider:
            case "gemini":
                return f"openai/{self.model}"
            case "groq":
                return f"groq/{self.model}"
            case "ollama":
                return f"ollama_chat/{self.model}"

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


MODELS: dict[str, ModelSpec] = {
    spec.alias: spec
    for spec in (
        ModelSpec(
            alias="gemini-3.8-flash",
            provider="gemini",
            model="gemini-3.8-flash",
            family="gemini",
            input_per_mtok=0.75,
            output_per_mtok=3.75,
            requests_per_minute=10,
            tokens_per_minute=250_000,
            label="Gemini 3.8 Flash",
            extra={"reasoning_effort": "low"},
        ),
        ModelSpec(
            alias="gemini-3.5-flash-lite",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            family="gemini",
            input_per_mtok=0.30,
            output_per_mtok=2.50,
            requests_per_minute=15,
            tokens_per_minute=250_000,
            label="Gemini 3.5 Flash-Lite",
        ),
        ModelSpec(
            alias="groq-qwen3.8-27b",
            provider="groq",
            model="qwen/qwen3.8-27b",
            family="qwen",
            input_per_mtok=0.80,
            output_per_mtok=4.00,
            requests_per_minute=30,
            tokens_per_minute=8_000,
            label="Qwen3.8 27B (Groq)",
            extra={"reasoning_effort": "none"},
        ),
        ModelSpec(
            alias="groq-gpt-oss-120b",
            provider="groq",
            model="openai/gpt-oss-120b",
            family="gpt-oss",
            input_per_mtok=0.15,
            output_per_mtok=0.60,
            requests_per_minute=30,
            tokens_per_minute=8_000,
            label="GPT-OSS 120B (Groq)",
            extra={"reasoning_effort": "low"},
        ),
        ModelSpec(
            alias="ollama-qwen3-8b",
            provider="ollama",
            model="qwen3:8b",
            family="qwen",
            # Self-hosted: no public list price. Priced at zero and labeled as such.
            input_per_mtok=0.0,
            output_per_mtok=0.0,
            requests_per_minute=600,
            tokens_per_minute=None,
            label="Qwen3 8B (local Ollama)",
            extra={"think": False},
        ),
    )
}

# Gemini 3.8 Flash allows only 20 free requests per day, so the default answer model is
# Qwen3.8 27B on Groq (1000/day). The judge stays a different family (GPT-OSS).
DEFAULT_ANSWER_MODEL = "groq-qwen3.8-27b"
DEFAULT_JUDGE_MODEL = "groq-gpt-oss-120b"


def get_model(alias: str) -> ModelSpec:
    try:
        return MODELS[alias]
    except KeyError:
        raise KeyError(f"Unknown model alias {alias!r}; known: {sorted(MODELS)}") from None
