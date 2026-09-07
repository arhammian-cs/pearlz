"""Centralized logging configuration for SupportPearlz.

Configures both console and rotating file handlers. Callers must never pass
API keys or other secrets into logged messages -- helper functions in this
module redact common secret-shaped substrings defensively.
"""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_-]{8,})")


class _RedactSecretsFilter(logging.Filter):
    """Strips substrings that look like API keys from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _SECRET_PATTERN.sub("sk-***REDACTED***", record.msg)
        if record.args:
            record.args = tuple(
                _SECRET_PATTERN.sub("sk-***REDACTED***", arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


def configure_logging(log_dir: Path, level: str = "INFO") -> None:
    """Idempotently configure root logging handlers for the application."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redact_filter = _RedactSecretsFilter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redact_filter)

    file_handler = RotatingFileHandler(
        log_dir / "supportpearlz.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redact_filter)

    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Keep third-party libraries from flooding the log at DEBUG level.
    for noisy_logger in ("httpx", "httpcore", "urllib3", "chromadb"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
