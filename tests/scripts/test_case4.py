"""Smoke tests for case4 CLI workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import case4


def _collect_records(output_dir: Path) -> list[dict]:
    records: list[dict] = []
    for file in output_dir.glob("case4_results_*.json"):
        with open(file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            records.extend(payload)
    return records


def test_case4_smoke_run_and_resume(tmp_path, monkeypatch) -> None:
    """Case4 should write results and skip duplicates when --resume is enabled."""
    output_dir = tmp_path / "case4"

    monkeypatch.setattr(
        "sys.argv",
        [
            "case4.py",
            "--datasets",
            "Wine",
            "--algorithms",
            "KNN",
            "--bits",
            "16",
            "--seeds",
            "42",
            "--output",
            str(output_dir),
            "--resume",
        ],
    )
    case4.main()
    first_records = _collect_records(output_dir)
    assert len(first_records) == 3

    case4.main()
    second_records = _collect_records(output_dir)
    # On resume, no new execution ids should be added.
    ids = [record.get("metadata", {}).get("execution_id") for record in second_records]
    assert len(ids) == len(set(ids))
