"""The LedgerLens agent as a LangGraph state machine.

    gather --(tool calls)--> tools --> gather ... --(READY or max rounds)--> answer
    answer --> verify --> self_check --(problems, one repair)--> answer
                                     \\--> finalize

- gather: the model plans and calls tools (single-turn native tool calling each round,
  so no provider has to replay hidden reasoning state across turns).
- answer: JSON answer that cites numbered evidence ([E3]) or refuses.
- verify: every number in the answer must be grounded in a passage, an XBRL fact or a
  calculate result (deterministic, no LLM call).
- self_check: every citation must point at real evidence and its quote must appear
  there; an answer left with no valid citation becomes a refusal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ledgerlens.agent import prompts
from ledgerlens.agent.tools import (
    EvidenceItem,
    EvidenceRegistry,
    RetrievalConfig,
    openai_tools,
    parse_arguments,
    run_tool,
)
from ledgerlens.llm import DEFAULT_ANSWER_MODEL, ToolCall, complete, track_usage
from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.schemas import AgentAnswer, AnswerUnit, Citation

StepCallback = Callable[[str, str], None]

REFUSAL_TEXT = "The filings I could retrieve do not support a reliable answer to this question."
_SCALES = {"one": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_UNITS: set[str] = {"USD", "shares", "percent", "ratio", "per_share", "count", "text"}


@dataclass(frozen=True)
class AgentConfig:
    model: str = DEFAULT_ANSWER_MODEL
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    max_rounds: int = 3
    xbrl_tool: bool = True
    verify_numbers: bool = True
    self_check: bool = True
    max_repairs: int = 1
    answer_evidence_chars: int = 14000
    temperature: float = 0.0


class State(TypedDict, total=False):
    question: str
    filters: SearchFilters
    round: int
    pending: list[ToolCall]
    draft: dict[str, Any]
    problems: list[str]
    repairs: int
    steps: list[str]


@dataclass
class _Run:
    """Per-question objects that are not part of the graph state."""

    config: AgentConfig
    registry: EvidenceRegistry
    on_step: StepCallback


# ------------------------------------------------------------------ helpers


def _evidence_brief(registry: EvidenceRegistry, chars: int = 300) -> str:
    if not registry.items:
        return "(none yet)"
    lines = []
    for item in registry.items.values():
        snippet = re.sub(r"\s+", " ", item.text)[:chars]
        lines.append(f"[{item.ref}] {item.header}: {snippet}")
    return "\n".join(lines)


def _evidence_full(registry: EvidenceRegistry, budget: int) -> str:
    # Facts and calculations first (short and exact), then passages in retrieval order.
    ordered = sorted(registry.items.values(), key=lambda i: {"fact": 0, "calc": 1}.get(i.kind, 2))
    parts, used = [], 0
    for item in ordered:
        block = f"[{item.ref}] {item.header}\n{item.text}"
        if used + len(block) > budget:
            block = block[: max(0, budget - used)]
        if not block:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _filters_line(filters: SearchFilters) -> str:
    active = {k: v for k, v in filters.model_dump().items() if v is not None}
    return f"Filters set by the user: {active}" if active else ""


def parse_json_answer(text: str) -> dict[str, Any]:
    """Parse the model's JSON answer, tolerating code fences and stray prose."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {"answer": text.strip(), "refused": False, "citations": []}


_NUM = re.compile(
    r"(?<![\w.])\(?-?\$?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\)?\s*"
    r"(trillion|billion|million|thousand|bn|mn|[BMK](?![a-zA-Z]))?",
)


def numbers_in(text: str) -> list[float]:
    """Numbers with any scale word applied. Skips citation refs like E3 and bare years."""
    text = re.sub(r"\[E\d+\]", " ", text)
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(1).replace(",", "")
        value = float(raw)
        scale = (m.group(2) or "").lower()
        if not scale and "." not in raw and 1900 <= value <= 2100:
            continue  # a year, not a quantity
        mult = {"trillion": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6,
                "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}.get(scale, 1.0)  # fmt: skip
        out.append(value * mult)
    return out


def _grounded(value: float, sources: list[float]) -> bool:
    """A number is grounded if some source number, at any reporting scale, matches it
    after rounding (0.5% relative), or it is a small count like 1-31."""
    if value == 0 or abs(value) <= 31:
        return True
    for src in sources:
        for mult in (1.0, 1e3, 1e6, 1e9):
            ref = src * mult
            if ref and abs(abs(value) - abs(ref)) / abs(ref) <= 0.005:
                return True
    return False


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9.%$]+", " ", text.lower()).strip()


def quote_supported(quote: str, text: str) -> bool:
    """Exact (normalized) substring match, or at least 80% of the quote's tokens present."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    tokens = q.split()
    present = sum(1 for tok in tokens if tok in t.split())
    return len(tokens) >= 3 and present / len(tokens) >= 0.8


# ------------------------------------------------------------------ graph


def build_graph(run: _Run) -> Any:
    cfg = run.config
    tool_names = ["search_filings", "get_table", "calculate"]
    if cfg.xbrl_tool:
        tool_names.insert(1, "get_company_facts")
    tools = openai_tools(tool_names)

    def gather(state: State) -> State:
        rnd = state.get("round", 0) + 1
        run.on_step("plan", f"round {rnd}")
        resp = complete(
            cfg.model,
            [
                {"role": "system", "content": prompts.GATHER_SYSTEM},
                {
                    "role": "user",
                    "content": prompts.GATHER_USER.format(
                        question=state["question"],
                        filters=_filters_line(state["filters"]),
                        evidence=_evidence_brief(run.registry),
                        round=rnd,
                        max_rounds=cfg.max_rounds,
                    ),
                },
            ],
            tools=tools,
            temperature=cfg.temperature,
            max_tokens=1024,
            purpose="gather",
        )
        allowed = set(tool_names)
        pending = [tc for tc in resp.tool_calls if tc.name in allowed][:8]
        return {"round": rnd, "pending": pending, "steps": [*state.get("steps", []), "plan"]}

    def call_tools(state: State) -> State:
        steps = list(state.get("steps", []))
        filters = state["filters"]
        for tc in state.get("pending", []):
            args = parse_arguments(tc.arguments)
            if tc.name == "search_filings":
                # User-selected filters always apply on top of what the model asked for.
                for key, value in filters.model_dump().items():
                    if value is not None:
                        args[key] = value
            result = run_tool(tc.name, args, run.registry, cfg.retrieval)
            run.on_step(tc.name, json.dumps(args)[:200])
            steps.append(tc.name)
            del result  # evidence is in the registry; the next gather round sees a brief
        return {"pending": [], "steps": steps}

    def after_gather(state: State) -> str:
        if state.get("pending") and state.get("round", 0) <= cfg.max_rounds:
            return "tools"
        return "answer"

    def after_tools(state: State) -> str:
        return "gather" if state.get("round", 0) < cfg.max_rounds else "answer"

    def answer(state: State) -> State:
        run.on_step("answer", "writing a cited answer")
        problems = state.get("problems") or []
        feedback = prompts.REPAIR_FEEDBACK.format(problems="; ".join(problems)) if problems else ""
        resp = complete(
            cfg.model,
            [
                {"role": "system", "content": prompts.ANSWER_SYSTEM},
                {
                    "role": "user",
                    "content": prompts.ANSWER_USER.format(
                        question=state["question"],
                        evidence=_evidence_full(run.registry, cfg.answer_evidence_chars)
                        or "(no evidence was found)",
                        feedback=feedback,
                    ),
                },
            ],
            temperature=cfg.temperature,
            max_tokens=1500,
            response_format={"type": "json_object"},
            purpose="answer" if not problems else "repair",
        )
        draft = parse_json_answer(resp.content)
        return {"draft": draft, "problems": [], "steps": [*state.get("steps", []), "answer"]}

    def verify(state: State) -> State:
        draft = state["draft"]
        if not cfg.verify_numbers or draft.get("refused"):
            return {}
        run.on_step("verify", "checking every number against the evidence")
        sources: list[float] = []
        for item in run.registry.items.values():
            sources.extend(numbers_in(item.text))
        bad = [v for v in numbers_in(str(draft.get("answer", ""))) if not _grounded(v, sources)]
        problems = list(state.get("problems") or [])
        if bad:
            shown = ", ".join(f"{v:,.6g}" for v in bad[:5])
            problems.append(f"these numbers do not appear in the evidence: {shown}")
        return {"problems": problems, "steps": [*state.get("steps", []), "verify"]}

    def self_check(state: State) -> State:
        draft = dict(state["draft"])
        steps = list(state.get("steps", []))
        problems = list(state.get("problems") or [])
        if not cfg.self_check:
            return {"draft": draft}
        run.on_step("self_check", "checking that every claim has a supporting source")
        valid = []
        for cit in draft.get("citations") or []:
            if not isinstance(cit, dict):
                continue
            item = run.registry.items.get(str(cit.get("ref", "")).strip("[] "))
            if item is None:
                problems.append(f"citation {cit.get('ref')} does not exist")
                continue
            quote = str(cit.get("quote", ""))
            if quote and not quote_supported(quote, item.text):
                problems.append(f"quote for {item.ref} is not in that source")
                continue
            valid.append(cit)
        cited_refs = set(re.findall(r"\[(E\d+)\]", str(draft.get("answer", ""))))
        known = {r for r in cited_refs if r in run.registry.items}
        if not draft.get("refused") and not valid and not known:
            problems.append("the answer cites no supporting evidence")
        draft["citations"] = valid
        steps.append("self_check")
        return {"draft": draft, "problems": problems, "steps": steps}

    def after_check(state: State) -> str:
        if state.get("problems") and state.get("repairs", 0) < cfg.max_repairs:
            return "repair"
        return END

    def repair(state: State) -> State:
        run.on_step("repair", "; ".join(state.get("problems") or [])[:200])
        return {"repairs": state.get("repairs", 0) + 1}

    graph = StateGraph(State)
    graph.add_node("gather", gather)
    graph.add_node("tools", call_tools)
    graph.add_node("answer", answer)
    graph.add_node("verify", verify)
    graph.add_node("self_check", self_check)
    graph.add_node("repair", repair)
    graph.add_edge(START, "gather")
    graph.add_conditional_edges("gather", after_gather, {"tools": "tools", "answer": "answer"})
    graph.add_conditional_edges("tools", after_tools, {"gather": "gather", "answer": "answer"})
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", "self_check")
    graph.add_conditional_edges("self_check", after_check, {"repair": "repair", END: END})
    # A repair re-answers with the problems as feedback (problems survive into answer).
    graph.add_edge("repair", "answer")
    return graph.compile()


# ------------------------------------------------------------------ finalize


def _to_citation(item: EvidenceItem, quote: str) -> Citation:
    return Citation(
        chunk_id=item.source_id,
        filing_id=item.filing_id or "",
        section=item.section,
        page=item.page,
        page_end=item.page_end,
        quote=quote[:300],
    )


def finalize(draft: dict[str, Any], registry: EvidenceRegistry, problems: list[str]) -> AgentAnswer:
    """Renumber [E#] markers to [1], [2] in order of first use and build Citation objects.

    Unresolved problems after the repair budget: an uncited answer becomes a refusal,
    anything else is kept (the eval scores it) with the problems in `steps`.
    """
    text = str(draft.get("answer") or "").strip()
    refused = bool(draft.get("refused"))
    quotes = {str(c.get("ref", "")).strip("[] "): str(c.get("quote", "")) for c in
              draft.get("citations") or [] if isinstance(c, dict)}  # fmt: skip
    order: list[str] = []
    for ref in re.findall(r"\[(E\d+)\]", text):
        item = registry.items.get(ref)
        if item is not None and item.kind != "calc" and ref not in order:
            order.append(ref)
    for ref in quotes:  # cited only in the citations list
        item = registry.items.get(ref)
        if item is not None and item.kind != "calc" and ref not in order:
            order.append(ref)
    if not refused and not order:
        refused, text = True, REFUSAL_TEXT
    number = {ref: i for i, ref in enumerate(order, start=1)}

    def renumber(m: re.Match[str]) -> str:
        n = number.get(m.group(1))
        return f"[{n}]" if n else ""

    text = re.sub(r"\[(E\d+)\]", renumber, text)
    text = re.sub(r"\s+([.,;])", r"\1", text).strip()
    citations = (
        [] if refused else [_to_citation(registry.items[r], quotes.get(r, "")) for r in order]
    )

    value: float | None = None
    unit = draft.get("unit") if draft.get("unit") in _UNITS else None
    if not refused and isinstance(draft.get("value"), int | float):
        scale = _SCALES.get(str(draft.get("scale") or "one").lower(), 1.0)
        value = float(draft["value"]) * (scale if unit in ("USD", "shares", "count") else 1.0)
    answer_unit: AnswerUnit | None = unit if value is not None else None  # type: ignore[assignment]
    return AgentAnswer(
        answer=text if not refused or text else REFUSAL_TEXT,
        refused=refused,
        answer_value=value,
        answer_unit=answer_unit,
        citations=citations,
        retrieved_chunk_ids=registry.chunk_ids(),
        steps=[f"unresolved: {p}" for p in problems],
    )


def run_agent(
    question: str,
    config: AgentConfig | None = None,
    filters: SearchFilters | None = None,
    on_step: StepCallback | None = None,
    registry: EvidenceRegistry | None = None,
) -> AgentAnswer:
    """Answer one question. Pass `registry` to inspect the evidence afterwards (the eval
    runner scores citations and retrieval against it)."""
    config = config or AgentConfig()
    registry = registry if registry is not None else EvidenceRegistry()
    run = _Run(config=config, registry=registry, on_step=on_step or (lambda name, detail: None))
    graph = build_graph(run)
    with track_usage() as usage:
        state = graph.invoke(
            {"question": question, "filters": filters or SearchFilters(), "steps": []},
            {"recursion_limit": 4 * config.max_rounds + 12},
        )
    result = finalize(state["draft"], registry, state.get("problems") or [])
    result.steps = [*state.get("steps", []), *result.steps]
    result.usage = usage.total
    return result
