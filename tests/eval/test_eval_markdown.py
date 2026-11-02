"""Markdown escaping for qids and names in tables and code spans."""

from __future__ import annotations

from ledgerlens.eval.markdown import md_cell, md_code


def test_md_cell_escapes_pipes_and_newlines() -> None:
    assert md_cell("a|b") == "a\\|b"
    assert md_cell("line one\n  line two") == "line one line two"
    assert md_cell("plain") == "plain"


def test_md_code_fences_around_backticks() -> None:
    assert md_code("q1") == "`q1`"
    # One inner backtick needs a two-backtick fence.
    assert md_code("q`1") == "``q`1``"
    # A run of two needs three.
    assert md_code("a``b") == "```a``b```"
    # Leading or trailing backticks are padded so the fence stays separate.
    assert md_code("`q") == "`` `q ``"
    assert md_code("x|y") == "`x\\|y`"
