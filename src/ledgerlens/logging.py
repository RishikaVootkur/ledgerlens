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
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
    _CONFIGURED = True


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
