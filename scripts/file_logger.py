#!/usr/bin/env python
"""Reusable file-logger helper for case study scripts."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def configure_case_file_logger(logger: logging.Logger, log_dir: str, file_prefix: str) -> Path:
    """Attach a single file handler to the provided logger and return the log path."""
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    os.makedirs(log_dir, exist_ok=True)
    log_file = Path(log_dir) / f"{file_prefix}_{_timestamp()}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return log_file
