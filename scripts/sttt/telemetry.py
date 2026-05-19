"""STTT telemetry helpers (one JSON file per run record)."""

from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def log_tse(record: dict[str, Any], *, output_dir: Path, phase: str) -> Path:
    """Write one crash-safe JSON file for an experiment record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    key = str(record.get("run_id") or record.get("run_key") or f"{phase}_{_stamp()}")
    safe_key = key.replace("/", "_")
    path = output_dir / f"{safe_key}.json"
    payload = {"timestamp": _stamp(), **record}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    return path


def load_table(path: Path) -> list[dict[str, Any]]:
    """Load all telemetry JSON records from a file or directory."""
    if not path.exists():
        return []
    if path.is_file():
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
            if not content:
                return []
            if content.startswith("{"):
                return [json.loads(content)]
            return [json.loads(line) for line in content.splitlines() if line.strip()]

    rows: list[dict[str, Any]] = []
    for json_file in sorted(glob.glob(str(path / "*.json"))):
        with open(json_file, "r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def is_done(*, output_dir: Path, phase: str, run_id: str) -> bool:
    """Check whether a run id has already been persisted."""
    _ = phase
    safe_key = run_id.replace("/", "_")
    path = output_dir / f"{safe_key}.json"
    return path.exists()
