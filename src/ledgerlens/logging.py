"""Structured logging with structlog. Never log secrets."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(stream=sys.stderr, level=level, format="%(message)s")
    renderer: Any = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        # Resolve sys.stderr on every call: test runners and CLIs swap the stream, and a
        # logger bound to a closed stream would fail later with "I/O on closed file".
        logger_factory=lambda *args: structlog.PrintLogger(sys.stderr),
    )
    _CONFIGURED = True


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
