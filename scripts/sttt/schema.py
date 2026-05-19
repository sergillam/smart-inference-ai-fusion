"""Unified schemas for STTT execution records."""

from __future__ import annotations

from typing import Any


def make_phase_record(
    *,
    run_id: str,
    dataset: str,
    model: str,
    seed: int,
    phase: str,
    wall_clock_ms: float,
    payload: dict[str, Any] | None = None,
    status: str = "success",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized phase record for telemetry."""
    row: dict[str, Any] = {
        "schema_version": "tse.v1",
        "run_id": run_id,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "phase": phase,
        "status": status,
        "wall_clock_ms": float(wall_clock_ms),
        "payload": payload or {},
    }
    if extra:
        row.update(extra)
    return row


def make_run_manifest(
    *,
    dataset: str,
    model: str,
    seed: int,
    phases: dict[str, dict[str, Any]],
    status: str = "success",
    error: str | None = None,
) -> dict[str, Any]:
    """Build one unified run-level payload with all phases."""
    manifest = {
        "schema_version": "tse.v1",
        "run_key": f"{dataset}:{model}:{seed}",
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "status": status,
        "phases": phases,
    }
    if error:
        manifest["error"] = error
    return manifest
