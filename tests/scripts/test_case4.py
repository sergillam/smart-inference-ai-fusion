"""Smoke tests for case4 CLI workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import case4

ROOT = Path(__file__).resolve().parents[2]


def _collect_records(output_dir: Path) -> list[dict]:
    records: list[dict] = []
    for file in output_dir.glob("case4_all_results_*.json"):
        with open(file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            records.extend(payload)
    return records


def test_case4_smoke_run_and_resume(tmp_path, monkeypatch) -> None:
    """Case4 should write results and skip duplicates when --resume is enabled."""
    output_dir = tmp_path / "case4"
    log_dir = tmp_path / "logs_case4"

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
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(log_dir),
            "--resume",
        ],
    )
    case4.main()
    first_records = _collect_records(output_dir)
    assert len(first_records) == 3
    assert list(output_dir.glob("case4_summary_*.json"))
    assert list(log_dir.glob("case4_*.log"))

    case4.main()
    second_records = _collect_records(output_dir)
    # On resume, no new execution ids should be added.
    ids = [record.get("metadata", {}).get("execution_id") for record in second_records]
    assert len(ids) == len(set(ids))


def test_case4_direct_script_entrypoint_help() -> None:
    """Direct script execution should resolve local imports correctly."""
    completed = subprocess.run(
        [sys.executable, "scripts/case4.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Case Study 4: SIP-Q Quantization" in completed.stdout
