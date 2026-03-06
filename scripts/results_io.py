"""Shared JSON I/O helpers for case study scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_records(results_dir: Path, pattern: str) -> list[dict[str, Any]]:
    """Load dict records from JSON list files matching a glob pattern."""
    records: list[dict[str, Any]] = []
    for file in results_dir.glob(pattern):
        try:
            with open(file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if isinstance(row, dict):
                records.append(row)
    return records
