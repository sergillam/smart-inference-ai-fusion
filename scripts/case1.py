#!/usr/bin/env python
"""
Case Study 1: SIP Architecture Validation

Objective: Validate the SIP architecture and measure the effect of synthetic
perturbations on classic supervised algorithms.

Datasets: Iris, Wine, Breast Cancer, Digits, Titanic
Algorithms: KNN, SVM, DT (Decision Tree), GNB (Gaussian Naive Bayes), MLP
Protocol: Two scenarios per algorithm and dataset - baseline (no perturbations)
          and SIP activated, repeated 5 times with distinct seeds.

Total experiments: 5 datasets × 5 algorithms × 2 scenarios × 5 seeds = 250 runs
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.experiments.common import run_impact_analysis, run_standard_experiment
from smart_inference_ai_fusion.models.gaussian_model import GaussianNBModel
from smart_inference_ai_fusion.models.knn_model import KNNModel
from smart_inference_ai_fusion.models.mlp_model import MLPModel
from smart_inference_ai_fusion.models.svm_model import SVMModel
from smart_inference_ai_fusion.models.tree_model import DecisionTreeModel
from smart_inference_ai_fusion.utils.types import (
    CSVDatasetName,
    DatasetSourceType,
    SklearnDatasetName,
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
    (DatasetSourceType.SKLEARN, SklearnDatasetName.IRIS, "Iris"),
    (DatasetSourceType.SKLEARN, SklearnDatasetName.WINE, "Wine"),
    (DatasetSourceType.SKLEARN, SklearnDatasetName.BREAST_CANCER, "Breast Cancer"),
    (DatasetSourceType.SKLEARN, SklearnDatasetName.DIGITS, "Digits"),
    (DatasetSourceType.CSV, CSVDatasetName.TITANIC, "Titanic"),
]

# Algorithms configuration with default parameters
ALGORITHMS: Dict[str, Tuple[Type[BaseModel], dict]] = {
    "KNN": (KNNModel, {"n_neighbors": 5}),
    "SVM": (SVMModel, {"kernel": "rbf", "C": 1.0, "random_state": None}),  # random_state set per seed
    "DT": (DecisionTreeModel, {"max_depth": 10, "random_state": None}),  # random_state set per seed
    "GNB": (GaussianNBModel, {}),
    "MLP": (MLPModel, {"hidden_layer_sizes": (100,), "max_iter": 500, "random_state": None}),  # random_state set per seed
}


def get_timestamp() -> str:
    """Get ISO-formatted timestamp for filenames."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def run_single_experiment(
    algorithm_name: str,
    model_class: Type[BaseModel],
    model_params: dict,
    dataset_source: DatasetSourceType,
    dataset_name,
    dataset_label: str,
    seed: int,
    impact_mode: bool = False,
) -> dict:
    """Run a single experiment with given configuration.

    Args:
        algorithm_name: Name of the algorithm (e.g., "KNN", "SVM")
        model_class: Model class to instantiate
        model_params: Parameters for the model
        dataset_source: Source type of the dataset
        dataset_name: Dataset identifier
        dataset_label: Human-readable dataset name
        seed: Random seed for reproducibility
        impact_mode: If True, run full impact analysis with isolated perturbations

    Returns:
        dict: Experiment results
    """
    # Update random_state in params if the model supports it
    params = model_params.copy()
    if "random_state" in params:
        params["random_state"] = seed

    mode_str = "(IMPACT ANALYSIS)" if impact_mode else ""
    logger.info(f"Running: {algorithm_name} on {dataset_label} with seed={seed} {mode_str}")

    try:
        if impact_mode:
            # P2: Run full impact analysis with isolated perturbations
            impact_results = run_impact_analysis(
                model_class=model_class,
                model_name=algorithm_name,
                dataset_source=dataset_source,
                dataset_name=dataset_name,
                model_params=params,
                seed=seed,
            )
            return {
                "status": "success",
                "algorithm": algorithm_name,
                "dataset": dataset_label,
                "seed": seed,
                "baseline_metrics": impact_results["experiments"]["baseline"],
                "inference_metrics": impact_results["experiments"]["all_combined"],
                "impact_analysis": impact_results["impact_analysis"],
                "isolated_experiments": {
                    "data_only": impact_results["experiments"]["data_only"],
                    "label_only": impact_results["experiments"]["label_only"],
                    "param_only": impact_results["experiments"]["param_only"],
                },
            }
        else:
            baseline_metrics, inference_metrics = run_standard_experiment(
                model_class=model_class,
                model_name=algorithm_name,
                dataset_source=dataset_source,
                dataset_name=dataset_name,
                model_params=params,
                seed=seed,
            )

            return {
                "status": "success",
                "algorithm": algorithm_name,
                "dataset": dataset_label,
                "seed": seed,
                "baseline_metrics": baseline_metrics,
                "inference_metrics": inference_metrics,
            }

    except Exception as e:
        logger.error(f"Error in {algorithm_name} on {dataset_label} (seed={seed}): {e}")
        return {
            "status": "error",
            "algorithm": algorithm_name,
            "dataset": dataset_label,
            "seed": seed,
            "error": str(e),
        }


def run_case_study_1(
    output_dir: str = "results/case1",
    algorithms: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    dry_run: bool = False,
    impact_mode: bool = False,
) -> dict:
    """Run Case Study 1: SIP Architecture Validation.

    Args:
        output_dir: Directory to save results
        algorithms: List of algorithm names to run (None = all)
        datasets: List of dataset names to run (None = all)
        seeds: List of seeds to use (None = all 5)
        dry_run: If True, only print what would be run
        impact_mode: If True, run P2 impact analysis with isolated perturbations

    Returns:
        dict: Summary of all experiment results
    """
    # Filter algorithms
    algo_list = algorithms or list(ALGORITHMS.keys())
    algo_configs = {k: v for k, v in ALGORITHMS.items() if k in algo_list}

    # Filter datasets
    dataset_list = datasets or [d[2] for d in DATASETS]
    dataset_configs = [d for d in DATASETS if d[2] in dataset_list]

    # Filter seeds
    seed_list = seeds or SEEDS

    # Calculate total experiments
    total_experiments = len(algo_configs) * len(dataset_configs) * len(seed_list)

    logger.info("=" * 60)
    logger.info("CASE STUDY 1: SIP ARCHITECTURE VALIDATION")
    if impact_mode:
        logger.info("MODE: IMPACT ANALYSIS (P2) - Isolated perturbation effects")
    logger.info("=" * 60)
    logger.info(f"Algorithms: {list(algo_configs.keys())}")
    logger.info(f"Datasets: {[d[2] for d in dataset_configs]}")
    logger.info(f"Seeds: {seed_list}")
    logger.info(f"Total experiments: {total_experiments}")
    if impact_mode:
        logger.info(f"  (5 sub-experiments each: baseline + 3 isolated + combined)")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN - No experiments will be executed")
        return {"dry_run": True, "total_experiments": total_experiments}

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Track results
    all_results = []
    summary = {
        "study": "Case Study 1 - SIP Architecture Validation",
        "timestamp": get_timestamp(),
        "configuration": {
            "algorithms": list(algo_configs.keys()),
            "datasets": [d[2] for d in dataset_configs],
            "seeds": seed_list,
            "total_experiments": total_experiments,
        },
        "results_by_algorithm": {},
        "results_by_dataset": {},
        "overall_stats": {
            "successful": 0,
            "failed": 0,
            "total_time_seconds": 0,
        },
    }

    start_time = time.time()
    experiment_count = 0

    for algo_name, (model_class, model_params) in algo_configs.items():
        summary["results_by_algorithm"][algo_name] = []

        for dataset_source, dataset_name, dataset_label in dataset_configs:
            if dataset_label not in summary["results_by_dataset"]:
                summary["results_by_dataset"][dataset_label] = []

            for seed in seed_list:
                experiment_count += 1
                logger.info(f"\n[{experiment_count}/{total_experiments}]")

                result = run_single_experiment(
                    algorithm_name=algo_name,
                    model_class=model_class,
                    model_params=model_params,
                    dataset_source=dataset_source,
                    dataset_name=dataset_name,
                    dataset_label=dataset_label,
                    seed=seed,
                    impact_mode=impact_mode,
                )

                all_results.append(result)
                summary["results_by_algorithm"][algo_name].append(result)
                summary["results_by_dataset"][dataset_label].append(result)

                if result["status"] == "success":
                    summary["overall_stats"]["successful"] += 1
                else:
                    summary["overall_stats"]["failed"] += 1

    summary["overall_stats"]["total_time_seconds"] = time.time() - start_time

    # Save all results
    results_file = os.path.join(output_dir, f"case1_all_results_{get_timestamp()}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"All results saved to: {results_file}")

    # Save summary
    summary_file = os.path.join(output_dir, f"case1_summary_{get_timestamp()}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to: {summary_file}")

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("CASE STUDY 1 COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Successful: {summary['overall_stats']['successful']}")
    logger.info(f"Failed: {summary['overall_stats']['failed']}")
    logger.info(f"Total time: {summary['overall_stats']['total_time_seconds']:.2f} seconds")
    logger.info("=" * 60)

    return summary


def main():
    """Main entry point for Case Study 1."""
    parser = argparse.ArgumentParser(
        description="Case Study 1: SIP Architecture Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python case1.py

  # Run only KNN and SVM on Iris dataset
  python case1.py --algorithms KNN SVM --datasets Iris

  # Run with specific seeds
  python case1.py --seeds 42 123

  # Dry run to see what would be executed
  python case1.py --dry-run

  # Run single experiment for quick test
  python case1.py --algorithms KNN --datasets Iris --seeds 42
        """,
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(ALGORITHMS.keys()),
        help="Algorithms to run (default: all)",
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
        "--output-dir",
        default="results/case1",
        help="Output directory for results (default: results/case1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )
    parser.add_argument(
        "--impact-analysis",
        action="store_true",
        help="Run P2 impact analysis mode (isolated perturbation effects)",
    )

    args = parser.parse_args()

    run_case_study_1(
        output_dir=args.output_dir,
        algorithms=args.algorithms,
        datasets=args.datasets,
        seeds=args.seeds,
        dry_run=args.dry_run,
        impact_mode=args.impact_analysis,
    )


if __name__ == "__main__":
    main()
