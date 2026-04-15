"""
logger.py — Centralised logging configuration.

Call ``get_logger(__name__)`` in any module to obtain a logger that writes to
both a rotating file and the console.

File handler: max 10 MB per file, 5 backups.
Console handler: same level as the root logger (configurable via LOG_LEVEL).
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5

_configured = False


def _configure_root(log_path: str = "./logs/bot.log", level: str = "INFO") -> None:
    """Configure the root logger once.  Subsequent calls are no-ops."""
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Rotating file handler ---
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initialising root handlers on first call.

    Uses LOG_PATH and LOG_LEVEL environment variables (with fallbacks) so
    this function works before a Config object is constructed.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance.
    """
    log_path = os.getenv("LOG_PATH", "./logs/bot.log")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    _configure_root(log_path=log_path, level=log_level)
    return logging.getLogger(name)
