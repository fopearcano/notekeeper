"""Logging configuration for Notekeeper.

A single ``configure_logging`` call wires up the root logger with a
human-readable console handler. Modules elsewhere in the app should obtain
their logger via ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"

_configured = False


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Optional[Path] = None,
) -> None:
    """Configure the root logger.

    Safe to call multiple times; subsequent calls are no-ops so test fixtures
    and the GUI entry point can both invoke it without duplicating handlers.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper so callers don't need to import ``logging`` too."""
    return logging.getLogger(name)
