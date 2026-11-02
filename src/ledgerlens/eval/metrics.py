"""Per-question scoring metrics.

Everything here is a pure function with no LLM calls, so each metric can be unit
tested against hand-computed values. The functions fall into five groups:

* Retrieval with binary relevance: `recall_at_k`, `hit_at_k`, `mrr`, `ndcg_at_k`.
* Relevance keys that turn labeled evidence and retrieved chunks into comparable
  strings: `evidence_keys`, `chunk_keys`, `citation_keys`.
* Numbers: `find_numbers`, `parse_number`, `extract_answer_value`, `numeric_match`,
  `numeric_correct`.
* Refusals: `is_refusal`, `refusal_correct`.
* Citations: `citation_precision`, `citation_recall`.

Metrics that do not apply to a question (for example recall when nothing is
labeled) return None. Aggregation code skips None values, so a missing label
never counts as a zero.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from ledgerlens.schemas import AgentAnswer, AnswerUnit, Chunk, Citation, Question

# ---------------------------------------------------------------- retrieval

RetrievedItem = str | Collection[str]
"""One ranked retrieval result.

Either a single id (a chunk id, a page key) or the set of relevance keys that one
retrieved chunk covers (see `chunk_keys`). A plain string is treated as the
one-element set containing it.
"""


def _as_keys(item: RetrievedItem) -> frozenset[str]:
    return frozenset((item,)) if isinstance(item, str) else frozenset(item)


def _gains(retrieved: Sequence[RetrievedItem], relevant: set[str], k: int) -> list[int]:
    """Binary gain per rank for the top `k` items.

    An item earns gain 1 when it covers at least one relevant key that no
    higher-ranked item has covered yet, else 0. This is how duplicates are
    handled: a repeated item (or a second chunk from an already matched page)
    keeps its rank slot but earns nothing, so only the first occurrence counts
    and no metric can exceed 1.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    covered: set[str] = set()
    gains: list[int] = []
    for item in retrieved[:k]:
        new = (_as_keys(item) & relevant) - covered
        gains.append(1 if new else 0)
        covered |= new
    return gains


def _covered(retrieved: Sequence[RetrievedItem], relevant: set[str], k: int) -> set[str]:
    covered: set[str] = set()
    for item in retrieved[:k]:
        covered |= _as_keys(item) & relevant
    return covered


def recall_at_k(retrieved: Sequence[RetrievedItem], relevant: set[str], k: int) -> float | None:
    """Fraction of relevant keys covered by the top `k` retrieved items.

    Returns None when `relevant` is empty: the question has no labeled evidence,
    so the caller should skip it rather than score it 0 or 1.
    """
    if not relevant:
        return None
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return len(_covered(retrieved, relevant, k)) / len(relevant)


def hit_at_k(retrieved: Sequence[RetrievedItem], relevant: set[str], k: int) -> float | None:
    """1.0 if any of the top `k` items is relevant, else 0.0. None if `relevant` is empty."""
    if not relevant:
        return None
    return 1.0 if any(_gains(retrieved, relevant, k)) else 0.0


def mrr(
    retrieved: Sequence[RetrievedItem], relevant: set[str], k: int | None = None
) -> float | None:
    """Reciprocal rank of the first relevant item (1-based), or 0.0 if none.

    `k` optionally truncates the list (MRR@k). Averaging this value over
    questions gives the mean reciprocal rank. None if `relevant` is empty.
    """
    if not relevant:
        return None
    gains = _gains(retrieved, relevant, max(1, len(retrieved)) if k is None else k)
    for rank, gain in enumerate(gains, start=1):
        if gain:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[RetrievedItem], relevant: set[str], k: int) -> float | None:
    """Normalized discounted cumulative gain with binary relevance.

    DCG@k = sum over ranks i = 1..k of gain_i / log2(i + 1), with the
    first-occurrence gains described in `_gains`. The ideal DCG puts a relevant
    item at each of the first min(k, |relevant|) ranks. Returns a value in
    [0, 1], or None if `relevant` is empty.

    Known limitation: one chunk that spans two relevant pages earns a single
    gain, so retrieving it alone scores below 1 even though nothing is missing.
    Recall@k does count both pages.
    """
    if not relevant:
        return None
    gains = _gains(retrieved, relevant, k)
    dcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal


# ---------------------------------------------------------------- relevance keys


def page_key(doc: str, page: int) -> str:
    """Key for one page of one document, e.g. `3M_2018_10K#p52`."""
    return f"{doc}#p{page}"


def evidence_keys(question: Question) -> set[str]:
    """Relevance keys for a question's labeled evidence, one rule per evidence row.

    Matching is by page where a page is labeled, because that is the finest unit
    both datasets share:

    * A row with a page gives `{doc}#p{page}`, where `doc` is the FinanceBench
      `doc_name` when set, else the `filing_id`. FinanceBench rows always land
      here.
    * A row with no page but `chunk_ids` (an XBRL fact located in the text)
      gives those chunk ids. Each listed chunk counts as its own relevant item,
      so the locator should list only the chunks that state the value.
    * A row with only a `filing_id` (an XBRL fact not yet located) gives the
      filing-level key `{filing_id}`. Any chunk from that filing then counts as
      relevant: coarse, but the only honest option without a page.

    Rows with no identifier are ignored. An empty result means retrieval and
    citation recall do not apply to this question.
    """
    keys: set[str] = set()
    for ev in question.evidence:
        doc = ev.doc_name or ev.filing_id
        if ev.page is not None and doc is not None:
            keys.add(page_key(doc, ev.page))
        elif ev.chunk_ids:
            keys.update(ev.chunk_ids)
        elif doc is not None:
            keys.add(doc)
    return keys


def _doc_and_page_keys(docs: list[str], page: int | None, page_end: int | None) -> set[str]:
    keys = set(docs)
    if page is not None:
        last = max(page, page_end if page_end is not None else page)
        for p in range(page, last + 1):
            keys.update(page_key(doc, p) for doc in docs)
    return keys


def chunk_keys(chunk: Chunk) -> set[str]:
    """Every relevance key a chunk can satisfy.

    The chunk id itself, the filing-level key `{filing_id}` (and `{doc_name}`
    when set), and a page key for every page from `page` to `page_end`
    inclusive, under both the filing id and the FinanceBench doc name. Emitting
    every identifier means a chunk matches evidence labeled any of the ways
    `evidence_keys` produces.
    """
    docs = [chunk.filing_id] + ([chunk.doc_name] if chunk.doc_name else [])
    return {chunk.chunk_id} | _doc_and_page_keys(docs, chunk.page, chunk.page_end)


def citation_keys(citation: Citation, chunk: Chunk | None = None) -> set[str]:
    """Relevance keys for a citation, for use with `citation_recall`.

    When the cited chunk is available its `chunk_keys` are used, which include
    the FinanceBench doc name. Otherwise the citation's own chunk id, filing id
    and page range are used; those cannot match evidence keyed only by
    `doc_name`, so pass the chunk when scoring FinanceBench.
    """
    if chunk is not None:
        return chunk_keys(chunk)
    keys = _doc_and_page_keys([citation.filing_id], citation.page, citation.page_end)
    return keys | {citation.chunk_id}


# ---------------------------------------------------------------- numbers

_SCALES: dict[str, float] = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
    "t": 1e12,
}

# The number itself. The lookbehind stops matches inside words and ids (Q3, FY2023,
# 1.2.3) and the lookahead skips form names such as 10-K and 10-Q.
_NUMBER = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?!\d)(?!-[KQ]\b)"
)
# What may sit immediately before a number: an accounting parenthesis, a currency
# marker and a sign, in either order ("-$5", "$-5", "($5", "$(5").
_PREFIX = re.compile(
    r"(?P<open>\()?\s*"
    r"(?:(?P<cur_a>US\$|(?<![A-Za-z])USD|\$|€|£)\s*)?"
    r"(?:(?<!\w)(?P<sign>[-−]))?\s*"
    r"(?:(?P<cur_b>US\$|(?<![A-Za-z])USD|\$)\s*)?"
    r"(?P<open2>\()?\s*$"
)
# What may follow: a closing parenthesis, a scale word, a percent marker, an "x".
_SUFFIX = re.compile(
    r"\s*(?P<close1>\))?"
    r"(?:\s*(?P<scale>thousand|million|billion|trillion|bn|mn|mm|tn|k|m|b|t)(?![A-Za-z]))?"
    r"(?:\s*(?P<pct>%|percent(?![a-z])|per\s?cent(?![a-z])|pct(?![a-z])))?"
    r"(?:\s*(?P<ratio>x|×)(?![A-Za-z]))?"
    r"\s*(?P<close2>\))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedNumber:
    """One number found in text, with the markers around it.

    `value` is the signed number as written, before any scale word is applied
    ("$1.2 billion" has value 1.2 and scale 1e9). `decimals` is how many digits
    follow the decimal point, which fixes the precision the text shows.
    """

    value: float
    scale: float
    decimals: int
    currency: bool
    percent: bool
    ratio: bool
    start: int
    end: int

    @property
    def base_value(self) -> float:
        """The value in base units, e.g. 1.2e9 for "$1.2 billion"."""
        return self.value * self.scale

    @property
    def marked(self) -> bool:
        """True when a currency, scale word, percent or ratio marker is attached."""
        return self.currency or self.percent or self.ratio or self.scale != 1.0

    @property
    def looks_like_year(self) -> bool:
        """A bare integer between 1900 and 2100, such as the 2023 in "fiscal 2023"."""
        return not self.marked and self.decimals == 0 and 1900 <= self.value <= 2100


def find_numbers(text: str) -> list[ParsedNumber]:
    """Every number in `text`, in reading order, with sign, scale and unit markers.

    Handles thousands separators, accounting negatives in parentheses
    ("(1,234)", "$(1.2) million"), leading minus signs (ASCII and Unicode),
    currency markers ($, US$, USD, €, £), scale words and suffixes (thousand, k,
    million, M, mn, MM, billion, B, bn, trillion, T), percent markers (%,
    percent, pct) and ratio suffixes (0.45x). A trailing sentence period is
    ignored. Thousands separators must be commas in groups of three.
    """
    found: list[ParsedNumber] = []
    for match in _NUMBER.finditer(text):
        raw = match.group(1)
        before = _PREFIX.search(text[max(0, match.start() - 8) : match.start()])
        after = _SUFFIX.match(text, match.end())
        assert before is not None and after is not None  # both patterns are all-optional
        value = float(raw.replace(",", ""))
        opened = bool(before.group("open") or before.group("open2"))
        closed = bool(after.group("close1") or after.group("close2"))
        if before.group("sign") or (opened and closed):
            value = -value
        scale_word = after.group("scale")
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        found.append(
            ParsedNumber(
                value=value,
                scale=_SCALES[scale_word.lower()] if scale_word else 1.0,
                decimals=decimals,
                currency=bool(before.group("cur_a") or before.group("cur_b")),
                percent=bool(after.group("pct")),
                ratio=bool(after.group("ratio")),
                start=match.start(),
                end=after.end(),
            )
        )
    return found


def _headline(numbers: list[ParsedNumber], unit: AnswerUnit | None = None) -> ParsedNumber | None:
    """Pick the number an answer is about.

    Heuristic, in order:

    1. If `unit` is "percent", the first number with a percent marker. If `unit`
       is "USD" or "per_share", the first number with a currency marker or scale.
    2. The first number with any marker (currency, scale, percent or ratio).
       Answers usually state the result with its unit and mention bare numbers
       (years, note numbers) only as context.
    3. The first number that does not look like a year (1900 to 2100, bare).
    4. The first number.
    """
    if not numbers:
        return None
    if unit == "percent":
        preferred = [n for n in numbers if n.percent]
    elif unit in ("USD", "per_share"):
        preferred = [n for n in numbers if n.currency or n.scale != 1.0]
    else:
        preferred = []
    for pool in (preferred, [n for n in numbers if n.marked]):
        if pool:
            return pool[0]
    not_years = [n for n in numbers if not n.looks_like_year]
    return not_years[0] if not_years else numbers[0]


def parse_number(text: str) -> float | None:
    """The headline number in `text` as written: signed, without its scale word.

    "$1,234.5 million" gives 1234.5, "(1,234)" gives -1234.0, "-12.3%" gives
    -12.3, "0.45x" gives 0.45. Returns None when the text has no number. Use
    `extract_answer_value` to get base units. See `_headline` for how one number
    is chosen when the text has several.
    """
    headline = _headline(find_numbers(text))
    return None if headline is None else headline.value


def extract_answer_value(answer_text: str, unit: AnswerUnit | None = None) -> float | None:
    """The headline number in base units, with scale words applied.

    "$1.2 billion" gives 1.2e9 and "USD 3.4B" gives 3.4e9. Percents stay in
    percent points ("12.5%" gives 12.5); `numeric_match` accepts a fraction
    against a percent. `unit` steers which number is chosen (see `_headline`).
    Returns None for unit "text" or when there is no number.
    """
    if unit == "text":
        return None
    headline = _headline(find_numbers(answer_text), unit)
    return None if headline is None else headline.base_value


def _rounding_half_step(gold: float, gold_text: str, unit: AnswerUnit | None) -> float:
    """Half of the last digit the gold text shows, in the units of `gold`.

    "$1.2 billion" shows tenths of a billion, so any prediction within 0.05e9 of
    the gold rounds to the same text. Returns 0 when the gold text has no
    number or its value does not correspond to `gold` (directly or as a percent
    stored as a fraction).
    """
    headline = _headline(find_numbers(gold_text), unit)
    if headline is None:
        return 0.0
    half_step = 0.5 * 10.0 ** (-headline.decimals) * headline.scale
    shown = headline.base_value
    if math.isclose(shown, gold, rel_tol=1e-9, abs_tol=1e-12):
        return half_step
    if unit == "percent" and math.isclose(shown, gold * 100, rel_tol=1e-9, abs_tol=1e-12):
        return half_step / 100
    return 0.0


def numeric_match(
    pred: float | None,
    gold: float,
    rel_tol: float,
    *,
    unit: AnswerUnit | None,
    abs_tol: float = 0.0,
    gold_text: str | None = None,
) -> bool:
    """Whether a predicted number matches the gold number.

    A prediction matches when it is within max(`rel_tol` * |gold|, `abs_tol`)
    of the gold (`abs_tol` is for golds near zero, where a relative tolerance
    collapses). Two allowances, and nothing else:

    * Percent as fraction: golds store percents as percent points (12.5 for
      12.5%). When `unit` is "percent" a prediction of 0.125 is the same answer,
      so the prediction is also tried multiplied and divided by 100.
    * Gold rounding: when `gold_text` is given, a prediction that rounds to the
      precision the gold text shows also matches. Gold "$1.2 billion" accepts
      1.24e9 because the gold itself is only stated to a tenth of a billion.

    Scale errors are never forgiven: 1.2e6 does not match 1.2e9, and a sign
    flip does not match either. A None or non-finite prediction never matches.
    """
    if pred is None or not math.isfinite(pred):
        return False
    tolerance = max(rel_tol * abs(gold), abs_tol)
    if gold_text is not None:
        tolerance = max(tolerance, _rounding_half_step(gold, gold_text, unit))
    # Absorb float noise, so a value exactly on the tolerance edge still matches.
    tolerance += 1e-9 * max(1.0, abs(gold))
    candidates = [pred, pred * 100, pred / 100] if unit == "percent" else [pred]
    return any(abs(c - gold) <= tolerance for c in candidates)


def numeric_correct(answer: AgentAnswer, question: Question) -> float | None:
    """Per-question numeric score: 1.0 if the answer's number matches the gold, else 0.0.

    Uses the agent's own `answer_value` when it set one, else the headline
    number parsed from its text. Tolerances, unit and gold text come from the
    question. None when the question has no numeric gold (text answers and
    unanswerable questions), so it is skipped rather than scored.
    """
    gold = question.answer_value
    if gold is None or question.answer_unit in (None, "text"):
        return None
    pred = answer.answer_value
    if pred is None:
        pred = extract_answer_value(answer.answer, question.answer_unit)
    matched = numeric_match(
        pred,
        gold,
        question.rel_tolerance,
        unit=question.answer_unit,
        abs_tol=question.abs_tolerance,
        gold_text=question.answer,
    )
    return 1.0 if matched else 0.0


# ---------------------------------------------------------------- refusals

_REFUSAL_PATTERNS = [
    r"(?:cannot|can ?not|can't|could not|couldn't) be (?:determined|answered|found|calculated)",
    r"(?:cannot|can't|could not|couldn't|unable to|not able to) "
    r"(?:determine|answer|find|locate|calculate|compute)",
    r"\bnot (?:been )?disclosed\b",
    r"\b(?:does|do|did) not (?:disclose|report|provide|break out)\b",
    r"\b(?:i|we) (?:do not|don't) have (?:enough|sufficient|the) (?:information|data)",
    r"\b(?:insufficient|not enough) (?:information|data|evidence)\b",
    r"\bfilings? (?:do|does|did) not (?:contain|include|provide|mention|report|disclose|state)",
    r"\bnot (?:available|found|provided|reported|mentioned|stated|included) in the "
    r"(?:provided |available |retrieved |relevant )?"
    r"(?:filings?|documents?|context|sources?|10-[kq]s?|reports?)",
    r"\bno (?:information|data|evidence) (?:is |was )?(?:available|provided|found)\b",
    r"\b(?:the )?(?:information|data) (?:is|was) not (?:available|provided)\b",
]
_REFUSAL = re.compile("|".join(f"(?:{p})" for p in _REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(answer_text: str) -> bool:
    """Whether the answer text declines to answer, judged by phrase patterns.

    Matches phrasings such as "cannot be determined from the filings", "not
    disclosed", "I don't have enough information" and "the filings do not
    contain". Curly apostrophes are normalized first. This is a fallback for
    answers that refuse in prose without setting `AgentAnswer.refused`; a real
    answer that mentions, say, an undisclosed side detail can trip it, so the
    agent's own flag remains the primary signal.
    """
    return bool(_REFUSAL.search(answer_text.replace("’", "'")))


def refusal_correct(answer: AgentAnswer, answerable: bool) -> bool:
    """True when the agent refused exactly when it should have.

    The agent counts as refusing if it set `refused` or its text reads as a
    refusal (`is_refusal`). Correct means refusing an unanswerable question or
    answering an answerable one. Whether the answer itself is right is scored
    separately.
    """
    refused = answer.refused or is_refusal(answer.answer)
    return refused != answerable


# ---------------------------------------------------------------- citations


def citation_precision(supported: list[bool]) -> float | None:
    """Fraction of citations that support the claim they are attached to.

    `supported` has one verdict per citation (from the judge or a string check).
    None when the answer has no citations, so an answer is not rewarded for
    citing nothing; count missing citations through `citation_recall`.
    """
    if not supported:
        return None
    return sum(supported) / len(supported)


def citation_recall(cited_keys: set[str], evidence_keys: set[str]) -> float | None:
    """Fraction of labeled evidence keys that the answer cites.

    Build `cited_keys` with `citation_keys` (or `chunk_keys` of the cited
    chunks) and `evidence_keys` with `evidence_keys(question)`, so both sides
    use the same page and filing keys. None when the question has no labeled
    evidence.
    """
    if not evidence_keys:
        return None
    return len(cited_keys & evidence_keys) / len(evidence_keys)
