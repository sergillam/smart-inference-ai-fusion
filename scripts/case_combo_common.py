#!/usr/bin/env python
"""Common helpers for combined SIP/SIP-V/SIP-Q case studies."""
# pylint: disable=wrong-import-position

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.case4_sip_q import (
    SEEDS,
    SUPERVISED_ALGOS,
    SUPERVISED_DATASETS,
    UNSUPERVISED_ALGOS,
    UNSUPERVISED_DATASETS,
    run_case_study_4,
)
from smart_inference_ai_fusion.experiments.common import run_standard_experiment
from smart_inference_ai_fusion.utils.verification_config import (
    SolverChoice,
    VerificationConfig,
    VerificationMode,
    set_verification_config,
)


def timestamp() -> str:
    """Return UTC timestamp suitable for filenames."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def configure_logger(
    logger_name: str, log_dir: str, file_prefix: str
) -> tuple[logging.Logger, Path]:
    """Create a logger that writes to stdout and a dedicated file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in logger.handlers:
            try:
                handler.close()
            except (OSError, ValueError):  # pragma: no cover - defensive cleanup path
                pass
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file = Path(log_dir) / f"{file_prefix}_{timestamp()}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_file


def configure_verification_runtime(enabled: bool, solver: str = "z3") -> None:
    """Configure global verification runtime used by the inference pipeline."""
    if not enabled:
        set_verification_config(
            VerificationConfig(mode=VerificationMode.INFERENCE, solver=SolverChoice.AUTO)
        )
        return

    solver_map = {
        "z3": SolverChoice.Z3,
        "cvc5": SolverChoice.CVC5,
        "both": SolverChoice.BOTH,
        "auto": SolverChoice.AUTO,
    }
    set_verification_config(
        VerificationConfig(
            mode=VerificationMode.VERIFICATION,
            solver=solver_map.get(solver.lower(), SolverChoice.Z3),
        )
    )


def run_sip_matrix(
    *,
    datasets: list[str],
    algorithms: list[str],
    seeds: list[int] | None = None,
    verification_enabled: bool = False,
    solver: str = "z3",
) -> list[dict[str, Any]]:
    """Run SIP matrix (with or without SIP-V verification) for selected configs."""
    seed_list = seeds or SEEDS
    configure_verification_runtime(verification_enabled, solver=solver)

    records: list[dict[str, Any]] = []
    for seed in seed_list:
        records.extend(
            _run_sip_group(
                seed=seed,
                datasets=datasets,
                algorithms=algorithms,
                dataset_map=SUPERVISED_DATASETS,
                algorithm_map=SUPERVISED_ALGOS,
                paradigm="supervised",
                verification_enabled=verification_enabled,
                solver=solver,
            )
        )
        records.extend(
            _run_sip_group(
                seed=seed,
                datasets=datasets,
                algorithms=algorithms,
                dataset_map=UNSUPERVISED_DATASETS,
                algorithm_map=UNSUPERVISED_ALGOS,
                paradigm="unsupervised",
                verification_enabled=verification_enabled,
                solver=solver,
            )
        )
    return records


def _run_sip_group(
    *,
    seed: int,
    datasets: list[str],
    algorithms: list[str],
    dataset_map: dict[str, tuple[Any, Any]],
    algorithm_map: dict[str, tuple[Any, dict[str, Any]]],
    paradigm: str,
    verification_enabled: bool,
    solver: str,
) -> list[dict[str, Any]]:
    """Run SIP/SIP-V experiments for a dataset/algorithm group."""
    group_records: list[dict[str, Any]] = []
    for dataset_label, (source, dataset_name) in dataset_map.items():
        if dataset_label not in datasets:
            continue
        for algo_key, (model_class, base_params) in algorithm_map.items():
            if algo_key not in algorithms:
                continue
            params = dict(base_params)
            if "random_state" in params:
                params["random_state"] = seed
            try:
                baseline, inference = run_standard_experiment(
                    model_class=model_class,
                    model_name=algo_key,
                    dataset_source=source,
                    dataset_name=dataset_name,
                    model_params=params,
                    seed=seed,
                )
                group_records.append(
                    {
                        "status": "success",
                        "dataset": dataset_label,
                        "algorithm": algo_key,
                        "seed": seed,
                        "paradigm": paradigm,
                        "verification_enabled": verification_enabled,
                        "solver": solver if verification_enabled else None,
                        "baseline_metrics": baseline,
                        "inference_metrics": inference,
                    }
                )
            except (ValueError, TypeError, RuntimeError, AttributeError) as exc:  # pragma: no cover
                group_records.append(
                    {
                        "status": "error",
                        "dataset": dataset_label,
                        "algorithm": algo_key,
                        "seed": seed,
                        "paradigm": paradigm,
                        "verification_enabled": verification_enabled,
                        "solver": solver if verification_enabled else None,
                        "error": str(exc),
                    }
                )
    return group_records


def run_quantization_matrix(
    *,
    output_dir: str,
    datasets: list[str],
    algorithms: list[str],
    bits: list[int],
    seeds: list[int] | None,
    method: str,
    dtype_profile: str,
    resume: bool,
) -> tuple[dict[str, Any], Path | None]:
    """Run Case4 quantization matrix and return summary + latest result path."""
    summary = run_case_study_4(
        output_dir=output_dir,
        datasets=datasets,
        algorithms=algorithms,
        bits=bits,
        seeds=seeds,
        method=method,
        dtype_profile=dtype_profile,
        resume=resume,
    )
    result_files = sorted(Path(output_dir).glob("case4_all_results_*.json"))
    latest_results = result_files[-1] if result_files else None
    return summary, latest_results


def save_combined_artifacts(
    *,
    output_dir: str,
    file_prefix: str,
    combination_name: str,
    quant_summary: dict[str, Any] | None,
    quant_results_file: Path | None,
    sip_records: list[dict[str, Any]] | None,
    notes: list[str] | None = None,
    elapsed_seconds: float | None = None,
) -> tuple[Path, Path]:
    """Persist combined results and summary artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    stamp = timestamp()
    all_results_file = Path(output_dir) / f"{file_prefix}_all_results_{stamp}.json"
    summary_file = Path(output_dir) / f"{file_prefix}_summary_{stamp}.json"

    all_payload = {
        "combination": combination_name,
        "timestamp": stamp,
        "quantization": {
            "summary": quant_summary,
            "all_results_file": str(quant_results_file) if quant_results_file else None,
        },
        "sip_or_sipv": {
            "records": sip_records or [],
        },
        "notes": notes or [],
    }

    success_count = len([r for r in (sip_records or []) if r.get("status") == "success"])
    error_count = len([r for r in (sip_records or []) if r.get("status") == "error"])
    summary_payload = {
        "combination": combination_name,
        "timestamp": stamp,
        "quant_records_generated": (
            (quant_summary or {}).get("overall_stats", {}).get("records_generated", 0)
        ),
        "sip_records_total": len(sip_records or []),
        "sip_records_success": success_count,
        "sip_records_error": error_count,
        "elapsed_seconds": elapsed_seconds,
        "notes": notes or [],
    }

    with open(all_results_file, "w", encoding="utf-8") as handle:
        json.dump(all_payload, handle, indent=2)
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    return all_results_file, summary_file


def default_case4_datasets() -> list[str]:
    """Return default dataset labels used by case4."""
    return ["Wine", "Digits", "MakeBlobs", "MakeMoons"]


def default_case4_algorithms() -> list[str]:
    """Return default algorithm keys used by case4."""
    return ["KNN", "DT", "MLP", "MBK", "GMM", "AC"]


def timed_run(fn: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """Execute a callable and return (result, elapsed_seconds)."""
    start = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start
