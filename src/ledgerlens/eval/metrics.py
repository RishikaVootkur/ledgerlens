"""Per-question scoring metrics.

Everything here is a pure function with no LLM calls, so each metric can be unit
tested against hand-computed values. The functions fall into five groups:

* Retrieval with binary relevance: `recall_at_k`, `hit_at_k`, `mrr`, `ndcg_at_k`.
* Relevance keys that turn labeled evidence and retrieved chunks into comparable
  strings: `evidence_keys`, `chunk_keys`, `citation_keys`.
* Numbers: `find_numbers`, `parse_number`, `extract_answer_value`, `numeric_match`,
  `numeric_match_kind`, `numeric_correct`.
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
from typing import Literal

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

_CURRENCY = r"US\$|(?<![A-Za-z])USD|\$|€|£"

# The number itself. The lookbehind stops matches inside words and ids (Q3, FY2023,
# 1.2.3), except right after "USD" ("USD3.4B"). The lookahead skips form names such
# as 10-K and 10-Q.
_NUMBER = re.compile(
    r"(?:(?<![\w.,])|(?<=USD))"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?!\d)(?!-[KQ]\b)"
)
# What may sit immediately before a number: an accounting parenthesis, a currency
# marker and a sign, in either order ("-$5", "$-5", "($5", "$(5").
_PREFIX = re.compile(
    r"(?P<open>\()?\s*"
    rf"(?:(?P<cur_a>{_CURRENCY})\s*)?"
    r"(?:(?<!\w)(?P<sign>[-−]))?\s*"
    rf"(?:(?P<cur_b>{_CURRENCY})\s*)?"
    r"(?P<open2>\()?\s*$"
)
# What may follow: a closing parenthesis, a scale word, a percent or basis-point
# marker, an "x". A one-letter scale (k, M, B, T) must not run into a letter, digit or
# apostrophe, so "3M's" is not 3 million.
_SUFFIX = re.compile(
    r"\s*(?P<close1>\))?"
    r"(?:\s*(?P<scale>(?:thousand|million|billion|trillion|bn|mn|mm|tn)(?![A-Za-z])"
    r"|[kmbt](?![\w'’])))?"
    r"(?:\s*(?:(?P<pct>%|percent(?![a-z])|per\s?cent(?![a-z])|pct(?![a-z]))"
    r"|(?P<bps>basis\s+points?|bps?)(?![A-Za-z])))?"
    r"(?:\s*(?P<ratio>x|×)(?![A-Za-z]))?"
    r"\s*(?P<close2>\))?",
    re.IGNORECASE,
)
# The text between two numbers that makes the second the end of a range:
# "10 - 12", "10-12%", "from $5 to $6 billion", "2019 through 2021".
_RANGE_GAP = re.compile(rf"\s*(?:[-–—]|to|through)\s*(?:(?:{_CURRENCY})\s*)?", re.IGNORECASE)
# Company names that start with digits. Without a currency marker these are names,
# never amounts ("3M had revenue of $32.8 billion").
_NAME_TOKENS = ("3M",)


@dataclass(frozen=True)
class ParsedNumber:
    """One number found in text, with the markers around it.

    `value` is the signed number as written, before any scale word is applied
    ("$1.2 billion" has value 1.2 and scale 1e9). `decimals` is how many digits
    follow the decimal point, which fixes the precision the text shows.
    `range_end` marks the second number of a range such as "from X to Y",
    "X to Y" or "X-Y".
    """

    value: float
    scale: float
    decimals: int
    currency: bool
    percent: bool
    ratio: bool
    start: int
    end: int
    range_end: bool = False

    @property
    def base_value(self) -> float:
        """The value in base units, e.g. 1.2e9 for "$1.2 billion" and 0.5 for "50 bps"."""
        return self.value * self.scale

    @property
    def marked(self) -> bool:
        """True when a currency, scale word, percent or ratio marker is attached."""
        return self.currency or self.percent or self.ratio or self.scale != 1.0

    @property
    def looks_like_year(self) -> bool:
        """A bare integer between 1900 and 2100, such as the 2023 in "fiscal 2023"."""
        return not self.marked and self.decimals == 0 and 1900 <= abs(self.value) <= 2100


def _is_name_token(text: str, start: int) -> bool:
    for token in _NAME_TOKENS:
        end = start + len(token)
        if text.startswith(token, start) and (end == len(text) or not text[end].isalnum()):
            return True
    return False


def find_numbers(text: str) -> list[ParsedNumber]:
    """Every number in `text`, in reading order, with sign, scale and unit markers.

    Handles:

    * thousands separators (commas in groups of three) and a trailing period;
    * accounting negatives in parentheses ("(1,234)", "$(1.2) million"), except a
      bare year such as "(2022)", which is a label and stays positive;
    * leading minus signs (ASCII and Unicode), except between two numbers: "10 - 12"
      and "10-12%" are ranges, not a negative 12;
    * currency markers ($, US$, USD, €, £), with or without a space ("USD3.4B");
    * scale words and suffixes (thousand, k, million, M, mn, MM, billion, B, bn,
      trillion, T); a one-letter suffix followed by a letter, digit or apostrophe
      is not a scale;
    * percent markers (%, percent, pct), basis points ("50 bps" is 0.5 percent)
      and ratio suffixes (0.45x).

    Skipped entirely: numbers glued to a word ("3M's", "5th", "2Q") and company
    names in `_NAME_TOKENS` ("3M") without a currency marker.

    Not handled: sign words. "A loss of $5 million" gives +5e6; the question
    decides whether a loss is negative, so the scorer cannot.
    """
    found: list[ParsedNumber] = []
    for match in _NUMBER.finditer(text):
        raw = match.group(1)
        before = _PREFIX.search(text[max(0, match.start() - 8) : match.start()])
        after = _SUFFIX.match(text, match.end())
        assert before is not None and after is not None  # both patterns are all-optional
        currency = bool(before.group("cur_a") or before.group("cur_b"))
        scale_word, pct, bps, ratio = (after.group(g) for g in ("scale", "pct", "bps", "ratio"))
        glued = match.end() < len(text) and text[match.end()].isalpha()
        if (glued and not (scale_word or pct or bps or ratio)) or (
            not currency and _is_name_token(text, match.start())
        ):
            continue
        prev = found[-1] if found else None
        range_end = prev is not None and bool(_RANGE_GAP.fullmatch(text, prev.end, match.start()))
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        bare_year = (
            not currency
            and not (scale_word or pct or bps or ratio)
            and decimals == 0
            and 1900 <= value <= 2100
        )
        opened = bool(before.group("open") or before.group("open2"))
        closed = bool(after.group("close1") or after.group("close2"))
        if (before.group("sign") and not range_end) or (opened and closed and not bare_year):
            value = -value
        if bps:
            scale = 0.01
        elif scale_word:
            scale = _SCALES[scale_word.lower()]
        else:
            scale = 1.0
        found.append(
            ParsedNumber(
                value=value,
                scale=scale,
                decimals=decimals,
                currency=currency,
                percent=bool(pct or bps),
                ratio=bool(ratio),
                start=match.start(),
                end=after.end(),
                range_end=range_end,
            )
        )
    return found


def _same_kind(first: ParsedNumber, second: ParsedNumber) -> bool:
    """Whether two numbers can be the two ends of one range.

    "12% to $5.6 billion" is a growth rate followed by an amount, not a range.
    A bare first number takes its unit from the second ("10-12%", "$5-6 billion").
    """
    if first.percent and not second.percent:
        return False
    return not (second.percent and first.marked and not first.percent)


def _headline(numbers: list[ParsedNumber], unit: AnswerUnit | None = None) -> ParsedNumber | None:
    """Pick the number an answer is about.

    Heuristic, in order:

    1. If `unit` is "percent", the first number with a percent marker. If `unit`
       is "USD" or "per_share", the first number with a currency marker, then the
       first with a scale word.
    2. The first number with any marker (currency, scale, percent or ratio).
       Answers usually state the result with its unit and mention bare numbers
       (years, note numbers) only as context.
    3. The first number that does not look like a year (1900 to 2100, bare).
    4. The first number.

    Then, if the pick is the start of a range ("from X to Y", "X to Y", "X-Y")
    with compatible units, the end of the range is taken instead. For a change
    ("margin fell from 30% to 25%") the end is the value the period closed at,
    which is what a question about the later period asks for. A question about
    the change itself should be answered with the change ("fell 5 points").
    """
    if not numbers:
        return None
    pools: list[list[ParsedNumber]] = []
    if unit == "percent":
        pools.append([n for n in numbers if n.percent])
    elif unit in ("USD", "per_share"):
        pools.append([n for n in numbers if n.currency])
        pools.append([n for n in numbers if n.scale != 1.0 and not n.percent])
    pools.append([n for n in numbers if n.marked])
    pools.append([n for n in numbers if not n.looks_like_year])
    pick = next((pool[0] for pool in pools if pool), numbers[0])
    i = numbers.index(pick)
    while i + 1 < len(numbers) and numbers[i + 1].range_end and _same_kind(pick, numbers[i + 1]):
        i += 1
        pick = numbers[i]
    return pick


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
    percent points ("12.5%" gives 12.5, "50 bps" gives 0.5). `unit` steers which
    number is chosen (see `_headline`). Returns None for unit "text" or when
    there is no number.
    """
    if unit == "text":
        return None
    headline = _headline(find_numbers(answer_text), unit)
    return None if headline is None else headline.base_value


def _has_percent_marker(text: str, unit: AnswerUnit | None) -> bool:
    headline = _headline(find_numbers(text), unit)
    return headline is not None and headline.percent


def _rounding_half_step(gold: float, gold_text: str, unit: AnswerUnit | None) -> float:
    """Half of the last digit the gold text shows, in base units.

    "$1.2 billion" shows tenths of a billion, so any prediction within 0.05e9 of
    the gold rounds to the same text. Returns 0 when the gold text has no
    number or its headline value is not the gold.
    """
    headline = _headline(find_numbers(gold_text), unit)
    if headline is None or not math.isclose(headline.base_value, gold, rel_tol=1e-9, abs_tol=1e-12):
        return 0.0
    return 0.5 * 10.0 ** (-headline.decimals) * abs(headline.scale)


MatchKind = Literal["tolerance", "fraction_as_percent", "gold_rounding"]

ROUNDING_CAP = 5.0
"""The gold-rounding allowance never exceeds this many times the relative tolerance."""


def numeric_match_kind(
    pred: float | None,
    gold: float,
    rel_tol: float,
    *,
    unit: AnswerUnit | None,
    abs_tol: float = 0.0,
    gold_text: str | None = None,
    pred_percent_marked: bool = False,
) -> MatchKind | None:
    """How a prediction matches the gold, or None when it does not.

    Checked in this order:

    1. "tolerance": within max(`rel_tol` * |gold|, `abs_tol`) of the gold.
       `abs_tol` is for golds near zero, where a relative tolerance collapses.
    2. "fraction_as_percent": golds store percents as percent points (12.5 for
       12.5%). When `unit` is "percent", the prediction carried no percent
       marker (`pred_percent_marked` is False) and |pred| <= 1.5, the
       prediction is read as a fraction and multiplied by 100, so 0.125
       matches 12.5. Never the other way: "100%" against a gold of 1.0 fails.
    3. "gold_rounding": with `gold_text`, a prediction that rounds to the
       precision the gold text shows, capped at `ROUNDING_CAP` times the
       relative tolerance. Gold "$1.2 billion" (rel_tol 1%) accepts 1.24e9,
       but gold "$1 billion" does not accept 1.49e9: its allowance is
       min(0.5e9, 5 * 1% * 1e9) = 0.05e9.

    Scale errors and sign flips are never forgiven. A None or non-finite
    prediction never matches. The kind lets reports count how many items pass
    only through an allowance.
    """
    if pred is None or not math.isfinite(pred):
        return None
    eps = 1e-9 * max(1.0, abs(gold))  # float noise on the tolerance edge
    tolerance = max(rel_tol * abs(gold), abs_tol)
    candidates: list[tuple[float, MatchKind]] = [(pred, "tolerance")]
    if unit == "percent" and not pred_percent_marked and abs(pred) <= 1.5:
        candidates.append((pred * 100, "fraction_as_percent"))
    for value, kind in candidates:
        if abs(value - gold) <= tolerance + eps:
            return kind
    if gold_text is not None:
        allowance = min(
            _rounding_half_step(gold, gold_text, unit), ROUNDING_CAP * rel_tol * abs(gold)
        )
        if any(abs(value - gold) <= max(tolerance, allowance) + eps for value, _ in candidates):
            return "gold_rounding"
    return None


def numeric_match(
    pred: float | None,
    gold: float,
    rel_tol: float,
    *,
    unit: AnswerUnit | None,
    abs_tol: float = 0.0,
    gold_text: str | None = None,
    pred_percent_marked: bool = False,
) -> bool:
    """Whether a predicted number matches the gold. See `numeric_match_kind` for the rules."""
    kind = numeric_match_kind(
        pred,
        gold,
        rel_tol,
        unit=unit,
        abs_tol=abs_tol,
        gold_text=gold_text,
        pred_percent_marked=pred_percent_marked,
    )
    return kind is not None


def numeric_correct(answer: AgentAnswer, question: Question) -> float | None:
    """Per-question numeric score: 1.0 if the answer's number matches the gold, else 0.0.

    Uses the agent's own `answer_value` when it set one, else the headline
    number parsed from its text. Whether the answer showed a percent marker is
    read from its text. Tolerances, unit and gold text come from the question.
    None when the question has no numeric gold (text answers and unanswerable
    questions), so it is skipped rather than scored.
    """
    gold = question.answer_value
    unit = question.answer_unit
    if gold is None or unit in (None, "text"):
        return None
    pred = answer.answer_value
    if pred is None:
        pred = extract_answer_value(answer.answer, unit)
    matched = numeric_match(
        pred,
        gold,
        question.rel_tolerance,
        unit=unit,
        abs_tol=question.abs_tolerance,
        gold_text=question.answer,
        pred_percent_marked=_has_percent_marker(answer.answer, unit),
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
