"""Escaping for user-controlled text (qids, metric and run names) in GitHub Markdown."""

from __future__ import annotations

import re


def md_cell(text: str) -> str:
    """Text safe inside a table cell: pipes escaped, line breaks flattened."""
    return re.sub(r"\s*[\r\n]+\s*", " ", text).replace("|", "\\|")


def md_code(text: str) -> str:
    """An inline code span that survives backticks and pipes in `text`.

    The fence is one backtick longer than the longest backtick run inside, and
    padded with spaces when the text starts or ends with a backtick (CommonMark
    strips one space on each side). Pipes are escaped because GitHub splits
    table cells on them even inside code spans.
    """
    text = md_cell(text)
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"
