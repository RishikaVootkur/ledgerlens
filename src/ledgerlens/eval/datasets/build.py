"""Regenerate the committed datasets. Order matters: smoke is drawn from xbrl."""

from __future__ import annotations

from collections.abc import Callable

from ledgerlens.eval.datasets import financebench, smoke, unanswerable, xbrl

GENERATORS: dict[str, Callable[[], int]] = {
    "financebench": financebench.generate,
    "xbrl": xbrl.generate,
    "unanswerable": unanswerable.generate,
    "smoke": smoke.generate,
}


def build_datasets(only: list[str] | None = None) -> dict[str, int]:
    """Run the selected generators (all by default) and return rows written per dataset.

    Rebuilding xbrl also rebuilds smoke, so the two never drift apart.
    """
    names = list(only or GENERATORS)
    unknown = set(names) - set(GENERATORS)
    if unknown:
        raise ValueError(f"unknown datasets {sorted(unknown)}; choose from {list(GENERATORS)}")
    if "xbrl" in names and "smoke" not in names:
        names.append("smoke")
    return {name: GENERATORS[name]() for name in GENERATORS if name in names}
