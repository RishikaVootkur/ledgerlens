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

from ledgerlens import tracing
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
    # Off only where no index exists (the scheduled live eval in CI uses XBRL + calculate).
    search_tool: bool = True
    verify_numbers: bool = True
    self_check: bool = True
    max_repairs: int = 1
    # Skip tool calls identical to one already executed for this question; if a planning
    # round proposes nothing new, go straight to answering. Off reproduces the original
    # behavior (models often re-issue the same XBRL lookup every round).
    dedupe_tool_calls: bool = True
    # Accept a number the model computed itself when it follows from two evidence numbers
    # by one arithmetic step (sum, difference, ratio, percent, percent change). Off
    # reproduces the original verifier, which refused such answers.
    derive_numbers: bool = True
    # ~3,500 tokens of numeric text: the answer prompt must fit one Groq free-tier
    # request (7,000 input tokens per minute for Qwen3.8 27B) with room for a repair.
    answer_evidence_chars: int = 10000
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
    executed: set[str] = field(default_factory=set)


# ------------------------------------------------------------------ helpers


def _call_key(tc: ToolCall) -> str:
    """Tool name plus arguments with key order and blank values normalized away."""
    args = {k: v for k, v in parse_arguments(tc.arguments).items() if v not in (None, "")}
    return f"{tc.name}:{json.dumps(args, sort_keys=True)}"


def _evidence_brief(registry: EvidenceRegistry, chars: int = 300, max_items: int = 18) -> str:
    """Short view of the evidence for planning rounds, capped so a gather prompt stays well
    under Groq's 8000 tokens per minute."""
    if not registry.items:
        return "(none yet)"
    lines = []
    for item in list(registry.items.values())[:max_items]:
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


_GROUPED_REFS = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)+)\s*\]")


def split_grouped_refs(text: str) -> str:
    """Models often group citations as [E1, E4]; rewrite them as [E1][E4] so every marker
    is seen (a grouped marker used to look like no citation at all and forced a refusal)."""
    return _GROUPED_REFS.sub(
        lambda m: "".join(f"[{r.strip()}]" for r in re.split(r"[,;]", m.group(1))), text
    )


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
    r"(?<![\w.])\(?-?(\$)?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\)?\s*"
    r"(trillion|billion|million|thousand|bn|mn|[BMK](?![a-zA-Z]))?",
)
_MULT = {"trillion": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6,
         "m": 1e6, "thousand": 1e3, "k": 1e3}  # fmt: skip
# Constants a calculation may use without evidence (percent, scale and period conversions).
_CALC_CONSTANTS = {1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 52.0, 100.0, 365.0, 1e3, 1e6, 1e9}


def numbers_in(text: str) -> list[float]:
    """Numbers with any scale word applied. Skips citation refs like E3 and bare years
    (a 4-digit integer from 1900 to 2100 with no $, comma, decimal or scale)."""
    text = re.sub(r"\[E\d+\]", " ", text)
    out = []
    for m in _NUM.finditer(text):
        dollar, raw_text, scale = m.group(1), m.group(2), (m.group(3) or "").lower()
        raw = raw_text.replace(",", "")
        value = float(raw)
        bare = not (dollar or scale or "," in raw_text or "." in raw)
        if bare and 1900 <= value <= 2100:
            continue
        out.append(value * _MULT.get(scale, 1.0))
    return out


def _matches(value: float, sources: list[float]) -> bool:
    """True if some source number, at any reporting scale (up or down), equals `value`
    within 0.5%: answers round, filings report in thousands or millions, and calculator
    inputs are usually written in the filing's own units."""
    for src in sources:
        for mult in (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9):
            ref = src * mult
            if ref and abs(abs(value) - abs(ref)) / abs(ref) <= 0.005:
                return True
    return False


def _is_small_count(value: float) -> bool:
    return value == int(value) and 0 <= value <= 12


def grounding_sources(registry: EvidenceRegistry) -> list[float]:
    """Numbers an answer may use: everything in passages and XBRL facts, plus the result of
    each calculation whose inputs are themselves grounded (so `calculate("523000 * 1")`
    cannot launder an invented number)."""
    base: list[float] = []
    for item in registry.items.values():
        if item.kind != "calc":
            base.extend(numbers_in(item.text))
    sources = list(base)
    for item in registry.items.values():
        if item.kind != "calc":
            continue
        expression, _, result = item.text.rpartition(" = ")
        inputs = numbers_in(expression)
        if all(v in _CALC_CONSTANTS or _matches(v, base) for v in inputs):
            sources.extend(numbers_in(result))
    return sources


def derivable(value: float, base: list[float], limit: int = 120) -> bool:
    """True if `value` is one arithmetic step from two evidence numbers: a+b, a-b, a/b,
    a/b*100 or (a-b)/b*100, with operands rescaled to a common reporting scale."""
    nums = sorted({abs(b) for b in base if b}, reverse=True)[:limit]
    target = abs(value)
    if target == 0:
        return False
    for i, a in enumerate(nums):
        for b in nums[i + 1 :] + nums[:i]:
            for scale in (1.0, 1e3, 1e6, 1e-3, 1e-6):
                bb = b * scale
                candidates = (
                    a + bb,
                    a - bb,
                    a / bb,
                    a / bb * 100,
                    (a - bb) / bb * 100,
                    (bb - a) / a * 100,
                    bb / a * 100,
                )
                for c in candidates:
                    c = abs(c)
                    if c and (abs(c - target) <= 0.005 * target or _matches(target, [c])):
                        return True
    return False


def ungrounded_numbers(answer: str, sources: list[float]) -> list[float]:
    return [v for v in numbers_in(answer) if not _is_small_count(v) and not _matches(v, sources)]


def _grounded(value: float, sources: list[float]) -> bool:
    return _is_small_count(value) or _matches(value, sources)


def _norm(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    return re.sub(r"[^a-z0-9.%$]+", " ", text).replace(". ", " ").strip(" .")


def quote_supported(quote: str, text: str) -> bool:
    """A quote supports a source when every number in it appears in the source and its
    words match: a normalized substring, or at least 80% of its tokens (3+ tokens)."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if not all(_matches(v, numbers_in(text)) or _is_small_count(v) for v in numbers_in(quote)):
        return False
    if q in t:
        return True
    tokens = [tok.strip(".") for tok in q.split()]
    vocab = {tok.strip(".") for tok in t.split()}
    present = sum(1 for tok in tokens if tok in vocab)
    return len(tokens) >= 3 and present / len(tokens) >= 0.8


# ------------------------------------------------------------------ graph


def _step_input(state: State) -> dict[str, Any]:
    """Small summary of what a step starts from, for the span's input.value."""
    summary: dict[str, Any] = {
        "question": state.get("question", "")[:300],
        "round": state.get("round", 0),
    }
    if state.get("pending"):
        summary["tool_calls"] = [
            {"name": tc.name, "arguments": tc.arguments[:300]} for tc in state["pending"]
        ]
    if state.get("problems"):
        summary["problems"] = state["problems"]
    if state.get("repairs"):
        summary["repairs"] = state["repairs"]
    return summary


def _traced(step: str, fn: Callable[[State], State]) -> Any:
    """Wrap a graph node in a CHAIN span named after the step (no-op when tracing is off)."""

    def node(state: State) -> State:
        if not tracing.enabled():
            return fn(state)
        attributes = {
            "ledgerlens.step": step,
            "ledgerlens.round": state.get("round", 0),
            tracing.INPUT_VALUE: tracing.to_json(_step_input(state)),
            tracing.INPUT_MIME: "application/json",
        }
        with tracing.span(step, "CHAIN", attributes) as span:
            out = fn(state)
            span.set_attribute(tracing.OUTPUT_VALUE, tracing.to_json(out))
            span.set_attribute(tracing.OUTPUT_MIME, "application/json")
            return out

    node.__name__ = fn.__name__
    return node


def build_graph(run: _Run) -> Any:
    cfg = run.config
    tool_names = ["search_filings", "get_table", "calculate"] if cfg.search_tool else ["calculate"]
    if cfg.xbrl_tool:
        # Position 1 when search is on keeps the original tool list (and every cached
        # request recorded with it) unchanged.
        tool_names.insert(1 if cfg.search_tool else 0, "get_company_facts")
    tools = openai_tools(tool_names)

    def gather(state: State) -> State:
        rnd = state.get("round", 0) + 1
        run.on_step("plan", f"round {rnd}")
        resp = complete(
            cfg.model,
            [
                {
                    "role": "system",
                    "content": prompts.GATHER_SYSTEM
                    if cfg.search_tool
                    else prompts.GATHER_SYSTEM_NO_SEARCH,
                },
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
        if cfg.dedupe_tool_calls:
            fresh = []
            for tc in pending:
                key = _call_key(tc)
                if key not in run.executed:
                    run.executed.add(key)
                    fresh.append(tc)
            pending = fresh
        return {"round": rnd, "pending": pending, "steps": [*state.get("steps", []), "plan"]}

    def call_tools(state: State) -> State:
        steps = list(state.get("steps", []))
        filters = state["filters"]
        for tc in state.get("pending", []):
            args = parse_arguments(tc.arguments)
            # User-selected filters always override what the model asked for.
            if tc.name == "search_filings":
                for key, value in filters.model_dump().items():
                    if value is not None:
                        args[key] = value
                if filters.fiscal_year is not None:
                    args["strict_year"] = True  # never relax a year the user chose
            elif tc.name == "get_company_facts":
                if filters.ticker:
                    args["ticker"] = filters.ticker
                if filters.fiscal_year is not None:
                    args["fiscal_year"] = filters.fiscal_year
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
        draft["answer"] = split_grouped_refs(str(draft.get("answer") or ""))
        return {"draft": draft, "problems": [], "steps": [*state.get("steps", []), "answer"]}

    def verify(state: State) -> State:
        draft = state["draft"]
        if not cfg.verify_numbers or draft.get("refused"):
            return {}
        run.on_step("verify", "checking every number against the evidence")
        sources = grounding_sources(run.registry)
        bad = ungrounded_numbers(str(draft.get("answer", "")), sources)
        if cfg.derive_numbers:
            bad = [v for v in bad if not derivable(v, sources)]
        problems = list(state.get("problems") or [])
        if bad:
            shown = ", ".join(f"{v:,.6g}" for v in bad[:5])
            problems.append(f"{NUMBERS}these numbers do not appear in the evidence: {shown}")
        return {"problems": problems, "steps": [*state.get("steps", []), "verify"]}

    def self_check(state: State) -> State:
        draft = dict(state["draft"])
        steps = list(state.get("steps", []))
        problems = list(state.get("problems") or [])
        if not cfg.self_check:
            return {"draft": draft}
        run.on_step("self_check", "checking that every claim has a supporting source")
        valid, invalid = check_citations(draft, run.registry, problems)
        text = strip_refs(str(draft.get("answer", "")), invalid)
        draft["answer"], draft["citations"] = text, valid
        cited = {r for r in re.findall(r"\[(E\d+)\]", text) if r in run.registry.items}
        if not draft.get("refused") and not (cited - {i for i in cited if _is_calc(run, i)}):
            problems.append(f"{NOCITE}the answer cites no supporting evidence")
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
    graph.add_node("gather", _traced("plan", gather))
    graph.add_node("tools", _traced("tools", call_tools))
    graph.add_node("answer", _traced("answer", answer))
    graph.add_node("verify", _traced("verify", verify))
    graph.add_node("self_check", _traced("self_check", self_check))
    graph.add_node("repair", _traced("repair", repair))
    graph.add_edge(START, "gather")
    graph.add_conditional_edges("gather", after_gather, {"tools": "tools", "answer": "answer"})
    graph.add_conditional_edges("tools", after_tools, {"gather": "gather", "answer": "answer"})
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", "self_check")
    graph.add_conditional_edges("self_check", after_check, {"repair": "repair", END: END})
    # A repair re-answers with the problems as feedback (problems survive into answer).
    graph.add_edge("repair", "answer")
    return graph.compile()


# ------------------------------------------------------------------ checks

NUMBERS = "numbers: "
NOCITE = "no citation: "


def _is_calc(run: _Run, ref: str) -> bool:
    item = run.registry.items.get(ref)
    return item is not None and item.kind == "calc"


def check_citations(
    draft: dict[str, Any], registry: EvidenceRegistry, problems: list[str]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Split the draft's citations into valid ones and refs to strip from the text.

    A passage citation needs a quote found in that passage (numbers included). XBRL facts
    and calculations may be cited without a quote. A ref used in the text with no valid
    citation entry is invalid too, so a dropped citation cannot survive as a bare marker.
    """
    valid: list[dict[str, Any]] = []
    invalid: set[str] = set()
    for cit in draft.get("citations") or []:
        if not isinstance(cit, dict):
            continue
        ref = str(cit.get("ref", "")).strip("[] ")
        item = registry.items.get(ref)
        if item is None:
            problems.append(f"citation {ref} does not exist")
            invalid.add(ref)
            continue
        quote = str(cit.get("quote") or "")
        if item.kind == "chunk" and not quote_supported(quote, item.text):
            problems.append(f"quote for {ref} is missing or not in that source")
            invalid.add(ref)
            continue
        valid.append({"ref": ref, "quote": quote})
    listed = {c["ref"] for c in valid}
    for ref in set(re.findall(r"\[(E\d+)\]", str(draft.get("answer", "")))):
        item = registry.items.get(ref)
        if item is None or (item.kind == "chunk" and ref not in listed):
            invalid.add(ref)
    return valid, invalid


def strip_refs(text: str, refs: set[str]) -> str:
    for ref in refs:
        text = text.replace(f"[{ref}]", "")
    return re.sub(r"\s+([.,;])", r"\1", text).strip()


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


def _as_number(value: object) -> float | None:
    """Models often return the value as a string ("21,155", "$34.5", "6.4%")."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[,$%\s]", "", value)
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def finalize(draft: dict[str, Any], registry: EvidenceRegistry, problems: list[str]) -> AgentAnswer:
    """Renumber [E#] markers to [1], [2] in order of first use and build Citation objects.

    When problems remain after the repair budget, an answer with ungrounded numbers or no
    supporting citation becomes a refusal: the agent prefers saying it cannot answer to
    shipping a number it could not tie to a source. Other leftovers are recorded in steps.
    """
    text = str(draft.get("answer") or "").strip()
    refused = draft.get("refused") is True
    if any(p.startswith((NUMBERS, NOCITE)) for p in problems):
        refused, text = True, REFUSAL_TEXT
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
    raw_value = _as_number(draft.get("value"))
    if refused:
        text = re.sub(r"\s*\[E?\d+\]", "", text).strip()
    if not refused and raw_value is not None:
        scale = _SCALES.get(str(draft.get("scale") or "one").lower(), 1.0)
        value = raw_value * (scale if unit in ("USD", "shares", "count") else 1.0)
        # Models sometimes write the full number and also name a scale ("359,241,000,000"
        # with "billion"). Prefer whichever reading the answer text itself states.
        stated = numbers_in(text)

        def said(x: float) -> bool:  # stated as-is, with no rescaling
            return any(abs(x - v) <= 0.005 * abs(v) for v in stated if v)

        if value != raw_value and not said(value) and said(raw_value):
            value = raw_value
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
    filters = filters or SearchFilters()
    attributes = {
        tracing.INPUT_VALUE: question,
        "ledgerlens.model_alias": config.model,
        "ledgerlens.filters": tracing.to_json(filters.model_dump(exclude_none=True)),
        "ledgerlens.retrieval.method": config.retrieval.method,
        "ledgerlens.xbrl_tool": config.xbrl_tool,
    }
    with tracing.span("agent", "AGENT", attributes) as span, track_usage() as usage:
        state = graph.invoke(
            {"question": question, "filters": filters, "steps": []},
            {"recursion_limit": 4 * config.max_rounds + 6 * (config.max_repairs + 1) + 10},
        )
        result = finalize(state["draft"], registry, state.get("problems") or [])
        result.steps = [*state.get("steps", []), *result.steps]
        result.usage = usage.total
        span.set_attributes(
            tracing.clean(
                {
                    tracing.OUTPUT_VALUE: result.answer,
                    "ledgerlens.refused": result.refused,
                    "ledgerlens.citations": [c.chunk_id for c in result.citations],
                    "ledgerlens.steps": result.steps,
                    "ledgerlens.usage.llm_calls": result.usage.llm_calls,
                    "ledgerlens.usage.prompt_tokens": result.usage.prompt_tokens,
                    "ledgerlens.usage.completion_tokens": result.usage.completion_tokens,
                    "ledgerlens.usage.cost_usd": result.usage.cost_usd,
                    "ledgerlens.cost_label": tracing.COST_LABEL,
                }
            )
        )
    return result
