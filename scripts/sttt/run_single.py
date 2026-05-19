"""Run one STTT configuration (dataset x model x seed)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from scripts.sttt.config import DATASETS, MODELS
from scripts.sttt.pandera_baseline import run_pandera_phase
from scripts.sttt.schema import make_phase_record, make_run_manifest
from scripts.sttt.telemetry import is_done, log_tse
from smart_inference_ai_fusion.experiments.common import run_impact_analysis
from smart_inference_ai_fusion.experiments.quantization_experiment import QuantizationExperiment
from smart_inference_ai_fusion.quantization.core.config import QuantizationConfig
from smart_inference_ai_fusion.utils.verification_config import (
    SolverChoice,
    VerificationConfig,
    VerificationMode,
    set_verification_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one STTT configuration")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", default="results/sttt")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _run_id(dataset_key: str, model_key: str, seed: int, phase: str) -> str:
    return f"{dataset_key}:{model_key}:{seed}:{phase}"


def _set_solver(solver: SolverChoice) -> None:
    set_verification_config(
        VerificationConfig(
            mode=VerificationMode.VERIFICATION,
            solver=solver,
        )
    )


def _model_params_for_dataset(dataset_key: str, model_key: str, base_params: dict[str, Any]) -> dict[str, Any]:
    """Apply dataset-specific fixed params required by STTT protocol."""
    params = dict(base_params)
    if dataset_key != "ieee":
        return params
    if model_key in {"lr", "dt"}:
        params["class_weight"] = "balanced"
    elif model_key == "rf":
        params["class_weight"] = "balanced_subsample"
    return params


def _extract_sipv_metrics(payload: dict[str, Any], solver_key: str) -> dict[str, Any]:
    """Extract compact SIP-V metrics from nested run_impact_analysis payload.

    Falls back to conservative defaults when fields are absent.
    """
    summary = (
        payload.get("experiments", {})
        .get("param_only", {})
        .get("verification_summary", {})
    )

    sections = [
        summary.get("pre_perturbation", {}).get(solver_key.upper(), {}),
        summary.get("post_perturbation", {}).get(solver_key.upper(), {}),
        summary.get("model_integrity", {}).get(solver_key.upper(), {}),
    ]
    sections = [s for s in sections if isinstance(s, dict)]

    execution_times = [float(s.get("execution_time", 0.0)) for s in sections]
    total_solver_s = sum(execution_times)
    status_values = [str(s.get("status", "")).upper() for s in sections]
    translation_ms = [float(s.get("translation_time_ms", 0.0) or 0.0) for s in sections]
    solve_ms = [float(s.get("solve_time_ms", 0.0) or 0.0) for s in sections]
    ram_mb = [float(s.get("peak_ram_mb", 0.0) or 0.0) for s in sections]
    num_constraints = [s.get("num_constraints") for s in sections if s.get("num_constraints") is not None]
    num_vars = [s.get("num_vars") for s in sections if s.get("num_vars") is not None]

    # Conservative split when translation time is not exported by the plugin yet.
    translation_s = sum(translation_ms) / 1000.0 if any(translation_ms) else 0.0
    solve_s = sum(solve_ms) / 1000.0 if any(solve_ms) else total_solver_s

    if any(status == "TIMEOUT" for status in status_values):
        final_status = "TIMEOUT"
    elif any(status == "ERROR" for status in status_values):
        final_status = "UNKNOWN"
    elif any(status == "FAILURE" for status in status_values):
        final_status = "SAT"
    elif all(status == "SUCCESS" for status in status_values if status):
        final_status = "UNSAT"
    else:
        final_status = "UNKNOWN"

    return {
        "status": final_status,
        "translation_time_ms": translation_s * 1000.0,
        "solve_time_ms": solve_s * 1000.0,
        "peak_ram_mb": max(ram_mb) if ram_mb else None,
        "num_constraints": max(num_constraints) if num_constraints else None,
        "num_vars": max(num_vars) if num_vars else None,
    }


def _run_baseline_phase(dataset_key: str, model_key: str, seed: int, output_dir: Path) -> dict[str, Any]:
    dataset = DATASETS[dataset_key]
    model = MODELS[model_key]
    model_params = _model_params_for_dataset(dataset_key, model_key, model.params)
    baseline_start = time.perf_counter()
    baseline = run_impact_analysis(
        model_class=model.model_class,
        model_name=model.key,
        dataset_source=dataset.source,
        dataset_name=dataset.name,
        model_params={**model_params, "random_state": seed},
        seed=seed,
    )
    baseline_ms = (time.perf_counter() - baseline_start) * 1000.0
    baseline_row = make_phase_record(
        run_id=_run_id(dataset_key, model_key, seed, "baseline"),
        dataset=dataset_key,
        model=model_key,
        seed=seed,
        phase="baseline",
        wall_clock_ms=baseline_ms,
        payload=baseline,
    )
    log_tse(baseline_row, output_dir=output_dir / "baseline", phase="records")
    return baseline_row


def _run_sip_family(dataset_key: str, model_key: str, seed: int, output_dir: Path) -> dict[str, Any]:
    dataset = DATASETS[dataset_key]
    model = MODELS[model_key]
    model_params = _model_params_for_dataset(dataset_key, model_key, model.params)

    sip_start = time.perf_counter()
    sip_result = run_impact_analysis(
        model_class=model.model_class,
        model_name=model.key,
        dataset_source=dataset.source,
        dataset_name=dataset.name,
        model_params={**model_params, "random_state": seed},
        seed=seed,
    )
    sip_ms = (time.perf_counter() - sip_start) * 1000.0
    sip_row = make_phase_record(
        run_id=_run_id(dataset_key, model_key, seed, "sip"),
        dataset=dataset_key,
        model=model_key,
        seed=seed,
        phase="sip",
        wall_clock_ms=sip_ms,
        payload=sip_result,
    )
    log_tse(sip_row, output_dir=output_dir / "sip", phase="records")
    phase_results: dict[str, Any] = {"sip": sip_row}

    for solver_name, solver in (("z3", SolverChoice.Z3), ("cvc5", SolverChoice.CVC5)):
        _set_solver(solver)
        start = time.perf_counter()
        sipv_payload = run_impact_analysis(
            model_class=model.model_class,
            model_name=model.key,
            dataset_source=dataset.source,
            dataset_name=dataset.name,
            model_params={**model_params, "random_state": seed},
            seed=seed,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        compact = _extract_sipv_metrics(sipv_payload, solver_name)

        sipv_row = make_phase_record(
            run_id=_run_id(dataset_key, model_key, seed, f"sipv_{solver_name}"),
            dataset=dataset_key,
            model=model_key,
            seed=seed,
            phase=f"sipv_{solver_name}",
            wall_clock_ms=elapsed_ms,
            payload=sipv_payload,
            extra={"solver": solver_name, **compact},
        )
        log_tse(sipv_row, output_dir=output_dir / "sipv", phase=solver_name)
        phase_results[f"sipv_{solver_name}"] = sipv_row

    return phase_results


def _run_sipq_phase(dataset_key: str, model_key: str, seed: int, output_dir: Path) -> list[dict[str, Any]]:
    dataset = DATASETS[dataset_key]
    model = MODELS[model_key]
    model_params = _model_params_for_dataset(dataset_key, model_key, model.params)
    config = QuantizationConfig(
        method="symmetric",
        data_bits=[8],
        model_bits=[8],
        enable_hybrid=True,
    )
    experiment = QuantizationExperiment(config)
    results = experiment.run_supervised(
        dataset_source=dataset.source,
        dataset_name=dataset.name,
        model_class=model.model_class,
        model_params={**model_params, "random_state": seed},
        seed=seed,
    )

    rows: list[dict[str, Any]] = []
    for idx, result in enumerate(results):
        payload = result.to_dict()
        meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        row = make_phase_record(
            run_id=_run_id(dataset_key, model_key, seed, f"sipq_{idx}"),
            dataset=dataset_key,
            model=model_key,
            seed=seed,
            phase="sipq",
            wall_clock_ms=float(payload.get("quantized_time_ms", 0.0) or 0.0),
            payload=payload,
            extra={
                "bit_width": payload.get("bit_width"),
                "decision_flip_rate": meta.get("decision_flip_rate"),
                "wasserstein_distance": meta.get("wasserstein_distance"),
                "f1_degradation": payload.get("f1_degradation"),
            },
        )
        log_tse(row, output_dir=output_dir / "sipq", phase="records")
        rows.append(row)
    return rows


def _run_pandera_phase(dataset_key: str, model_key: str, seed: int, output_dir: Path) -> dict[str, Any]:
    dataset = DATASETS[dataset_key]
    model = MODELS[model_key]
    model_params = _model_params_for_dataset(dataset_key, model_key, model.params)
    result = run_pandera_phase(
        dataset_source=dataset.source,
        dataset_name=dataset.name,
        model_params={**model_params, "random_state": seed},
        seed=seed,
    )
    row = make_phase_record(
        run_id=_run_id(dataset_key, model_key, seed, "pandera"),
        dataset=dataset_key,
        model=model_key,
        seed=seed,
        phase="pandera",
        wall_clock_ms=result.wall_clock_ms,
        payload={},
        extra={
            "caught_fault": result.caught_fault,
            "num_failures": result.num_failures,
            "failures_by_type": result.failures_by_type,
            "dependency_available": result.dependency_available,
        },
    )
    log_tse(row, output_dir=output_dir / "pandera", phase="records")
    return row


def main() -> None:
    args = _parse_args()
    dataset_key = args.dataset.lower()
    model_key = args.model.lower()

    if dataset_key not in DATASETS:
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Available: {sorted(DATASETS.keys())}")
    if model_key not in MODELS:
        raise ValueError(f"Unsupported model '{args.model}'. Available: {sorted(MODELS.keys())}")

    output_dir = Path(args.results_dir or args.output_dir)

    phases = ["baseline", "sip", "sipv_z3", "sipv_cvc5", "pandera"]
    if args.skip_existing and all(
        is_done(
            output_dir=output_dir / ("sipv" if "sipv" in phase else phase.split("_")[0]),
            phase=(phase.split("_")[-1] if "sipv" in phase else "records"),
            run_id=_run_id(dataset_key, model_key, args.seed, phase),
        )
        for phase in phases
    ):
        return

    phase_results: dict[str, Any] = {}
    run_status = "success"
    run_error: str | None = None
    start = time.perf_counter()

    try:
        phase_results["baseline"] = _run_baseline_phase(dataset_key, model_key, args.seed, output_dir)
        phase_results.update(_run_sip_family(dataset_key, model_key, args.seed, output_dir))
        phase_results["sipq"] = _run_sipq_phase(dataset_key, model_key, args.seed, output_dir)
        phase_results["pandera"] = _run_pandera_phase(dataset_key, model_key, args.seed, output_dir)
    except Exception as exc:
        run_status = "failed"
        run_error = f"{exc.__class__.__name__}: {exc}"
        phase_results["run_error"] = make_phase_record(
            run_id=_run_id(dataset_key, model_key, args.seed, "run_error"),
            dataset=dataset_key,
            model=model_key,
            seed=args.seed,
            phase="run_error",
            wall_clock_ms=(time.perf_counter() - start) * 1000.0,
            payload={},
            status="failed",
            extra={"error": run_error},
        )

    run_manifest = make_run_manifest(
        dataset=dataset_key,
        model=model_key,
        seed=args.seed,
        phases=phase_results,
        status=run_status,
        error=run_error,
    )
    log_tse(run_manifest, output_dir=output_dir / "runs", phase="records")

    if run_status != "success":
        raise RuntimeError(run_error or "run failed")


if __name__ == "__main__":
    main()
