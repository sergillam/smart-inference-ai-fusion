#!/usr/bin/env python
"""
Case Study 3: SIP-V Formal Verification Evaluation

Objective: Empirically evaluate the SIP-V verifiable extension integrated into SIP.
Three central aspects are quantified:
    1. Efficacy of the verification mechanism in distinguishing success/failure scenarios
    2. Computational cost introduced by formal solvers
    3. Comparative behavior of Z3 and CVC5 backends on identical verification tasks

Datasets: Wine, Make Moons, Make Blobs
Models: Decision Tree, Logistic Regression, MLP Classifier, MiniBatch KMeans

Protocol: Each model-dataset combination is executed with:
    - No verification (baseline timing)
    - Z3 verification enabled
    - CVC5 verification enabled
    Repeated 5 times with distinct seeds.

Experiment pairs: 3 datasets × 4 models × 3 verification modes × 5 seeds = 180 runs
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.experiments.common import (
    run_baseline_experiment,
    run_inference_experiment,
)

# Models
from smart_inference_ai_fusion.models.logistic_regression_model import LogisticRegressionModel
from smart_inference_ai_fusion.models.minibatch_kmeans_model import MiniBatchKMeansModel
from smart_inference_ai_fusion.models.mlp_model import MLPModel
from smart_inference_ai_fusion.models.tree_model import DecisionTreeModel
from smart_inference_ai_fusion.utils.types import (
    DatasetSourceType,
    SklearnDatasetName,
)
from smart_inference_ai_fusion.utils.verification_config import (
    SolverChoice,
    VerificationConfig,
    VerificationMode,
    set_verification_config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# STUDY CONFIGURATION
# =============================================================================

# Seeds for reproducibility (5 distinct seeds)
SEEDS = [42, 123, 456, 789, 1024]

# Datasets configuration
DATASETS = [
    (DatasetSourceType.SKLEARN, SklearnDatasetName.WINE, "Wine", 3),
    (DatasetSourceType.SKLEARN, SklearnDatasetName.MAKE_MOONS, "Make Moons", 2),
    (DatasetSourceType.SKLEARN, SklearnDatasetName.MAKE_BLOBS, "Make Blobs", 3),
]

# Models configuration
MODELS: Dict[str, Tuple[Type[BaseModel], dict, str]] = {
    "DT": (DecisionTreeModel, {"max_depth": 10, "random_state": None}, "supervised"),
    "LR": (LogisticRegressionModel, {"max_iter": 1000, "random_state": None}, "supervised"),
    "MLP": (
        MLPModel,
        {"hidden_layer_sizes": (100,), "max_iter": 500, "random_state": None},
        "supervised",
    ),
    "MBK": (
        MiniBatchKMeansModel,
        {"n_clusters": None, "random_state": None, "batch_size": 256},
        "unsupervised",
    ),
}

# Verification modes configuration
# The VerificationConfig will be set globally before each experiment
VERIFICATION_MODES = {
    "none": VerificationConfig(mode=VerificationMode.BASIC, solver=SolverChoice.AUTO),
    "z3": VerificationConfig(mode=VerificationMode.VERIFICATION, solver=SolverChoice.Z3),
    "cvc5": VerificationConfig(mode=VerificationMode.VERIFICATION, solver=SolverChoice.CVC5),
}


def get_timestamp() -> str:
    """Get ISO-formatted timestamp for filenames."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def run_single_verification_experiment(
    model_name: str,
    model_class: Type[BaseModel],
    model_params: dict,
    model_type: str,
    dataset_source: DatasetSourceType,
    dataset_name: SklearnDatasetName,
    dataset_label: str,
    n_clusters: int,
    seed: int,
    verification_mode: str,
    verification_config: VerificationConfig,
) -> dict:
    """Run a single verification experiment.

    Args:
        model_name: Short name of the model (e.g., "DT", "LR")
        model_class: Model class to instantiate
        model_params: Parameters for the model
        model_type: "supervised" or "unsupervised"
        dataset_source: Source type of the dataset
        dataset_name: Dataset identifier
        dataset_label: Human-readable dataset name
        n_clusters: Number of clusters (for unsupervised models)
        seed: Random seed for reproducibility
        verification_mode: "none", "z3", or "cvc5"
        verification_config: Verification configuration to set globally

    Returns:
        dict: Experiment results including verification metrics
    """
    # Prepare parameters
    params = model_params.copy()
    if "random_state" in params:
        params["random_state"] = seed
    if model_type == "unsupervised" and "n_clusters" in params:
        params["n_clusters"] = n_clusters

    mode_label = verification_mode.upper() if verification_mode != "none" else "NO-VERIF"
    logger.info(f"Running: {model_name} on {dataset_label} | Mode: {mode_label} | seed={seed}")

    # Set the global verification config BEFORE running the experiment
    set_verification_config(verification_config)

    try:
        # Measure total execution time
        start_time = time.perf_counter()

        if verification_mode == "none":
            # Run baseline without verification for timing comparison
            metrics = run_baseline_experiment(
                model_class=model_class,
                model_name=model_name,
                dataset_source=dataset_source,
                dataset_name=dataset_name,
                filtered_params=params,
                seed=seed,
            )
            verification_results = None
        else:
            # Run with verification enabled (config already set globally)
            metrics = run_inference_experiment(
                model_class=model_class,
                model_name=model_name,
                dataset_source=dataset_source,
                dataset_name=dataset_name,
                filtered_params=params,
                seed=seed,
            )
            # Extract verification results if available
            verification_results = metrics.pop("verification_summary", None)

        total_time = time.perf_counter() - start_time

        return {
            "status": "success",
            "model": model_name,
            "model_type": model_type,
            "dataset": dataset_label,
            "seed": seed,
            "verification_mode": verification_mode,
            "total_execution_time_ms": total_time * 1000,
            "metrics": metrics,
            "verification_results": verification_results,
        }

    except Exception as e:
        logger.error(
            f"Error in {model_name} on {dataset_label} (mode={verification_mode}, seed={seed}): {e}"
        )
        return {
            "status": "error",
            "model": model_name,
            "model_type": model_type,
            "dataset": dataset_label,
            "seed": seed,
            "verification_mode": verification_mode,
            "error": str(e),
        }


def compute_verification_statistics(results: List[dict]) -> dict:
    """Compute aggregate statistics from verification experiments.

    Args:
        results: List of experiment result dictionaries

    Returns:
        dict: Aggregate statistics
    """
    stats = {
        "by_solver": {"none": [], "z3": [], "cvc5": []},
        "by_model": {},
        "by_dataset": {},
    }

    for r in results:
        if r["status"] != "success":
            continue

        mode = r["verification_mode"]
        model = r["model"]
        dataset = r["dataset"]
        exec_time = r.get("total_execution_time_ms", 0)

        # Collect times by solver
        stats["by_solver"][mode].append(exec_time)

        # Collect by model
        if model not in stats["by_model"]:
            stats["by_model"][model] = {"none": [], "z3": [], "cvc5": []}
        stats["by_model"][model][mode].append(exec_time)

        # Collect by dataset
        if dataset not in stats["by_dataset"]:
            stats["by_dataset"][dataset] = {"none": [], "z3": [], "cvc5": []}
        stats["by_dataset"][dataset][mode].append(exec_time)

    # Compute summary statistics
    def summarize(times: List[float]) -> dict:
        if not times:
            return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "p95": 0}
        arr = np.array(times)
        return {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    summary = {
        "solver_comparison": {mode: summarize(times) for mode, times in stats["by_solver"].items()},
        "overhead": {},
        "by_model": {},
        "by_dataset": {},
    }

    # Compute overhead (verification time - baseline time)
    baseline_mean = summary["solver_comparison"]["none"]["mean"]
    for mode in ["z3", "cvc5"]:
        mode_mean = summary["solver_comparison"][mode]["mean"]
        if baseline_mean > 0:
            overhead_ms = mode_mean - baseline_mean
            overhead_pct = (overhead_ms / baseline_mean) * 100
        else:
            overhead_ms = mode_mean
            overhead_pct = 0
        summary["overhead"][mode] = {
            "absolute_ms": overhead_ms,
            "relative_pct": overhead_pct,
        }

    # Per-model statistics
    for model, mode_times in stats["by_model"].items():
        summary["by_model"][model] = {mode: summarize(times) for mode, times in mode_times.items()}

    # Per-dataset statistics
    for dataset, mode_times in stats["by_dataset"].items():
        summary["by_dataset"][dataset] = {
            mode: summarize(times) for mode, times in mode_times.items()
        }

    return summary


def extract_verification_efficacy(results: List[dict]) -> dict:
    """Extract verification efficacy metrics (SAT/UNSAT counts).

    Args:
        results: List of experiment result dictionaries

    Returns:
        dict: Verification efficacy statistics
    """
    efficacy = {
        "z3": {"sat": 0, "unsat": 0, "error": 0, "total_constraints": 0},
        "cvc5": {"sat": 0, "unsat": 0, "error": 0, "total_constraints": 0},
    }

    for r in results:
        mode = r.get("verification_mode")
        if mode not in ["z3", "cvc5"]:
            continue

        vr = r.get("verification_results")
        if not vr or not isinstance(vr, dict):
            continue

        # The verification results are nested by phase and solver
        # Structure: {phase: {solver: {constraints_satisfied: [], constraints_violated: []}}}
        solver_key = "Z3" if mode == "z3" else "CVC5"

        for phase in ["pre_perturbation", "post_perturbation", "model_integrity"]:
            phase_data = vr.get(phase, {})
            if not isinstance(phase_data, dict):
                continue

            solver_data = phase_data.get(solver_key, {})
            if not isinstance(solver_data, dict):
                continue

            satisfied = len(solver_data.get("constraints_satisfied", []))
            violated = len(solver_data.get("constraints_violated", []))
            efficacy[mode]["unsat"] += satisfied  # UNSAT = constraint holds
            efficacy[mode]["sat"] += violated  # SAT = constraint violated
            efficacy[mode]["total_constraints"] += satisfied + violated

    return efficacy


def generate_latex_tables(summary: dict, efficacy: dict) -> str:
    """Generate LaTeX tables from experiment summary.

    Args:
        summary: Aggregate statistics
        efficacy: Verification efficacy data

    Returns:
        str: LaTeX formatted tables
    """
    latex = []

    # Table 1: Solver Comparison (Execution Time)
    latex.append(
        r"""
% Table 1: Computational Cost Comparison
\begin{table}[htbp]
\centering
\caption{Custo Computacional por Solver (tempo em ms)}
\label{tab:solver_cost}
\begin{tabular}{lrrrr}
\toprule
\textbf{Modo} & \textbf{Média} & \textbf{Desvio} & \textbf{P95} & \textbf{Overhead} \\
\midrule"""
    )

    for mode in ["none", "z3", "cvc5"]:
        s = summary["solver_comparison"].get(mode, {})
        overhead = summary["overhead"].get(mode, {})
        overhead_str = f"+{overhead.get('relative_pct', 0):.1f}\\%" if mode != "none" else "--"
        latex.append(
            f"{mode.upper():8} & {s.get('mean', 0):.2f} & {s.get('std', 0):.2f} & "
            f"{s.get('p95', 0):.2f} & {overhead_str} \\\\"
        )

    latex.append(
        r"""\bottomrule
\end{tabular}
\end{table}
"""
    )

    # Table 2: Verification Efficacy
    latex.append(
        r"""
% Table 2: Verification Efficacy
\begin{table}[htbp]
\centering
\caption{Eficácia da Verificação Formal}
\label{tab:verification_efficacy}
\begin{tabular}{lrrr}
\toprule
\textbf{Solver} & \textbf{Válidos (UNSAT)} & \textbf{Violados (SAT)} & \textbf{Total} \\
\midrule"""
    )

    for solver in ["z3", "cvc5"]:
        e = efficacy.get(solver, {})
        latex.append(
            f"{solver.upper():6} & {e.get('unsat', 0):4d} & {e.get('sat', 0):4d} & "
            f"{e.get('total_constraints', 0):4d} \\\\"
        )

    latex.append(
        r"""\bottomrule
\end{tabular}
\end{table}
"""
    )

    # Table 3: Per-Model Breakdown
    latex.append(
        r"""
% Table 3: Per-Model Execution Time
\begin{table}[htbp]
\centering
\caption{Tempo de Execução por Modelo (média em ms)}
\label{tab:model_time}
\begin{tabular}{lrrr}
\toprule
\textbf{Modelo} & \textbf{Baseline} & \textbf{Z3} & \textbf{CVC5} \\
\midrule"""
    )

    for model in ["DT", "LR", "MLP", "MBK"]:
        model_stats = summary["by_model"].get(model, {})
        none_mean = model_stats.get("none", {}).get("mean", 0)
        z3_mean = model_stats.get("z3", {}).get("mean", 0)
        cvc5_mean = model_stats.get("cvc5", {}).get("mean", 0)
        latex.append(f"{model:8} & {none_mean:.2f} & {z3_mean:.2f} & {cvc5_mean:.2f} \\\\")

    latex.append(
        r"""\bottomrule
\end{tabular}
\end{table}
"""
    )

    return "\n".join(latex)


def run_case_study_3(
    output_dir: str = "results/case3",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    verification_modes: Optional[List[str]] = None,
    dry_run: bool = False,
) -> dict:
    """Run Case Study 3: SIP-V Formal Verification Evaluation.

    Args:
        output_dir: Directory to save results
        models: List of model names to run (None = all)
        datasets: List of dataset names to run (None = all)
        seeds: List of seeds to use (None = all 5)
        verification_modes: List of modes to run ("none", "z3", "cvc5")
        dry_run: If True, only print what would be run

    Returns:
        dict: Summary of all experiment results
    """
    # Filter models
    model_list = models or list(MODELS.keys())
    model_configs = {k: v for k, v in MODELS.items() if k in model_list}

    # Filter datasets
    dataset_list = datasets or [d[2] for d in DATASETS]
    dataset_configs = [d for d in DATASETS if d[2] in dataset_list]

    # Filter seeds
    seed_list = seeds or SEEDS

    # Filter verification modes
    mode_list = verification_modes or list(VERIFICATION_MODES.keys())

    # Calculate total experiments
    total_experiments = len(model_configs) * len(dataset_configs) * len(seed_list) * len(mode_list)

    logger.info("=" * 70)
    logger.info("CASE STUDY 3: SIP-V FORMAL VERIFICATION EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Models: {list(model_configs.keys())}")
    logger.info(f"Datasets: {[d[2] for d in dataset_configs]}")
    logger.info(f"Verification modes: {mode_list}")
    logger.info(f"Seeds: {seed_list}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info("=" * 70)

    if dry_run:
        logger.info("DRY RUN - No experiments will be executed")
        logger.info("\nExperiments that would be run:")
        for model_name in model_configs.keys():
            for _, _, dataset_label, _ in dataset_configs:
                for mode in mode_list:
                    logger.info(f"  {model_name} on {dataset_label} | {mode.upper()}")
        return {"dry_run": True, "total_experiments": total_experiments}

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Track results
    all_results = []
    summary = {
        "study": "Case Study 3 - SIP-V Formal Verification Evaluation",
        "timestamp": get_timestamp(),
        "configuration": {
            "models": list(model_configs.keys()),
            "datasets": [d[2] for d in dataset_configs],
            "verification_modes": mode_list,
            "seeds": seed_list,
            "total_experiments": total_experiments,
        },
        "results_by_mode": {mode: [] for mode in mode_list},
        "results_by_model": {},
        "results_by_dataset": {},
        "overall_stats": {
            "successful": 0,
            "failed": 0,
            "total_time_seconds": 0,
        },
    }

    start_time = time.time()
    experiment_count = 0

    for model_name, (model_class, model_params, model_type) in model_configs.items():
        if model_name not in summary["results_by_model"]:
            summary["results_by_model"][model_name] = []

        for dataset_source, dataset_name, dataset_label, n_clusters in dataset_configs:
            if dataset_label not in summary["results_by_dataset"]:
                summary["results_by_dataset"][dataset_label] = []

            for seed in seed_list:
                for mode in mode_list:
                    experiment_count += 1
                    logger.info(f"\n[{experiment_count}/{total_experiments}]")

                    verification_config = VERIFICATION_MODES[mode]

                    result = run_single_verification_experiment(
                        model_name=model_name,
                        model_class=model_class,
                        model_params=model_params,
                        model_type=model_type,
                        dataset_source=dataset_source,
                        dataset_name=dataset_name,
                        dataset_label=dataset_label,
                        n_clusters=n_clusters,
                        seed=seed,
                        verification_mode=mode,
                        verification_config=verification_config,
                    )

                    all_results.append(result)
                    summary["results_by_mode"][mode].append(result)
                    summary["results_by_model"][model_name].append(result)
                    summary["results_by_dataset"][dataset_label].append(result)

                    if result["status"] == "success":
                        summary["overall_stats"]["successful"] += 1
                    else:
                        summary["overall_stats"]["failed"] += 1

    summary["overall_stats"]["total_time_seconds"] = time.time() - start_time

    # Compute statistics
    logger.info("\nComputing verification statistics...")
    verification_stats = compute_verification_statistics(all_results)
    verification_efficacy = extract_verification_efficacy(all_results)

    summary["verification_statistics"] = verification_stats
    summary["verification_efficacy"] = verification_efficacy

    # Generate LaTeX tables
    latex_tables = generate_latex_tables(verification_stats, verification_efficacy)
    summary["latex_tables"] = latex_tables

    # Save all results
    results_file = os.path.join(output_dir, f"case3_all_results_{get_timestamp()}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"All results saved to: {results_file}")

    # Save summary
    summary_file = os.path.join(output_dir, f"case3_summary_{get_timestamp()}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to: {summary_file}")

    # Save LaTeX tables
    latex_file = os.path.join(output_dir, f"case3_latex_tables_{get_timestamp()}.tex")
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_tables)
    logger.info(f"LaTeX tables saved to: {latex_file}")

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("CASE STUDY 3 COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Successful: {summary['overall_stats']['successful']}")
    logger.info(f"Failed: {summary['overall_stats']['failed']}")
    logger.info(f"Total time: {summary['overall_stats']['total_time_seconds']:.2f} seconds")
    logger.info("\n--- SOLVER COMPARISON (mean execution time) ---")
    for mode, stats in verification_stats["solver_comparison"].items():
        logger.info(f"  {mode.upper():8}: {stats['mean']:.2f} ms (±{stats['std']:.2f})")
    logger.info("\n--- VERIFICATION EFFICACY ---")
    for solver in ["z3", "cvc5"]:
        e = verification_efficacy[solver]
        logger.info(f"  {solver.upper()}: {e['unsat']} valid (UNSAT), {e['sat']} violated (SAT)")
    logger.info("=" * 70)

    return summary


def main():
    """Main entry point for Case Study 3."""
    parser = argparse.ArgumentParser(
        description="Case Study 3: SIP-V Formal Verification Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python case3_sip_v.py

  # Run only Z3 verification
  python case3_sip_v.py --modes z3

  # Compare Z3 vs CVC5 only (skip baseline)
  python case3_sip_v.py --modes z3 cvc5

  # Run specific models
  python case3_sip_v.py --models DT LR

  # Run with specific datasets
  python case3_sip_v.py --datasets Wine "Make Moons"

  # Dry run to see what would be executed
  python case3_sip_v.py --dry-run

  # Quick test
  python case3_sip_v.py --models DT --datasets Wine --seeds 42 --modes none z3
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[d[2] for d in DATASETS],
        help="Datasets to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help=f"Seeds for reproducibility (default: {SEEDS})",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(VERIFICATION_MODES.keys()),
        help="Verification modes to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/case3",
        help="Output directory for results (default: results/case3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )

    args = parser.parse_args()

    run_case_study_3(
        output_dir=args.output_dir,
        models=args.models,
        datasets=args.datasets,
        seeds=args.seeds,
        verification_modes=args.modes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
