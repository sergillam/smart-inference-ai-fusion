#!/usr/bin/env python
"""
Case Study 2: Paradigm Comparison - Supervised vs Unsupervised Robustness

Objective: Compare paradigms and investigate the effect of labels and
architectural redundancy on robustness.

Datasets:
    - Classification (supervised): Wine, Digits
    - Clustering/High dimensionality (unsupervised): 20 Newsgroups, Make Moons, LFW

Algorithms:
    - Supervised: KNN, SVM, DT (Decision Tree), GNB (Gaussian Naive Bayes), MLP
    - Unsupervised: MBK (MiniBatchKMeans), GMM (Gaussian Mixture), AC (Agglomerative),
                    ICA (FastICA), SC (Spectral Clustering)

Protocol: Two scenarios per algorithm and dataset - baseline (no perturbations)
          and SIP activated, repeated 5 times with distinct seeds.

Experiment pairs: (2 datasets × 5 supervised + 3 datasets × 5 unsupervised) × 5 seeds
                = (10 + 15) × 5 = 125 experiment pairs (baseline + SIP)
Total scenario runs: 125 pairs × 2 scenarios = 250 runs
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.artifact_relocation import relocate_new_default_artifacts, snapshot_default_artifacts
from scripts.file_logger import configure_case_file_logger
from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.experiments.common import (
    run_impact_analysis,
    run_standard_experiment,
)

# Unsupervised models
from smart_inference_ai_fusion.models.agglomerative_clustering_model import (
    AgglomerativeClusteringModel,
)
from smart_inference_ai_fusion.models.fastica_model import FastICAModel
from smart_inference_ai_fusion.models.gaussian_mixture_model import GaussianMixtureModel

# Supervised models
from smart_inference_ai_fusion.models.gaussian_model import GaussianNBModel
from smart_inference_ai_fusion.models.knn_model import KNNModel
from smart_inference_ai_fusion.models.minibatch_kmeans_model import MiniBatchKMeansModel
from smart_inference_ai_fusion.models.mlp_model import MLPModel
from smart_inference_ai_fusion.models.spectral_clustering_model import SpectralClusteringModel
from smart_inference_ai_fusion.models.svm_model import SVMModel
from smart_inference_ai_fusion.models.tree_model import DecisionTreeModel
from smart_inference_ai_fusion.utils.types import (
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

# Supervised datasets (classification)
SUPERVISED_DATASETS = [
    (DatasetSourceType.SKLEARN, SklearnDatasetName.WINE, "Wine"),
    (DatasetSourceType.SKLEARN, SklearnDatasetName.DIGITS, "Digits"),
]

# Unsupervised datasets (clustering / high dimensionality)
UNSUPERVISED_DATASETS = [
    (DatasetSourceType.SKLEARN, SklearnDatasetName.NEWSGROUPS_20, "20 Newsgroups"),
    (DatasetSourceType.SKLEARN, SklearnDatasetName.MAKE_MOONS, "Make Moons"),
    (DatasetSourceType.SKLEARN, SklearnDatasetName.LFW_PEOPLE, "LFW People"),
]

# Supervised algorithms configuration
SUPERVISED_ALGORITHMS: Dict[str, Tuple[Type[BaseModel], dict]] = {
    "KNN": (KNNModel, {"n_neighbors": 5}),
    "SVM": (SVMModel, {"kernel": "rbf", "C": 1.0, "random_state": None}),
    "DT": (DecisionTreeModel, {"max_depth": 10, "random_state": None}),
    "GNB": (GaussianNBModel, {}),
    "MLP": (MLPModel, {"hidden_layer_sizes": (100,), "max_iter": 500, "random_state": None}),
}

# Unsupervised algorithms configuration
# n_clusters will be set dynamically based on dataset
UNSUPERVISED_ALGORITHMS: Dict[str, Tuple[Type[BaseModel], dict]] = {
    "MBK": (MiniBatchKMeansModel, {"n_clusters": None, "random_state": None, "batch_size": 256}),
    "GMM": (GaussianMixtureModel, {"n_components": None, "random_state": None, "max_iter": 100}),
    "AC": (AgglomerativeClusteringModel, {"n_clusters": None, "linkage": "ward"}),
    "ICA": (FastICAModel, {"n_components": None, "random_state": None, "max_iter": 200}),
    "SC": (
        SpectralClusteringModel,
        {"n_clusters": None, "random_state": None, "affinity": "nearest_neighbors"},
    ),
}

# Default number of clusters per dataset
DATASET_N_CLUSTERS = {
    "20 Newsgroups": 20,  # 20 newsgroups categories
    "Make Moons": 2,  # 2 moons
    "LFW People": 10,  # Use 10 clusters for face recognition
}


def get_timestamp() -> str:
    """Get ISO-formatted timestamp for filenames."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def run_single_experiment(
    algorithm_name: str,
    model_class: Type[BaseModel],
    model_params: dict,
    dataset_source: DatasetSourceType,
    dataset_name: Union[SklearnDatasetName, str],
    dataset_label: str,
    seed: int,
    paradigm: str,
    impact_mode: bool = False,
) -> dict:
    """Run a single experiment with given configuration.

    Args:
        algorithm_name: Name of the algorithm (e.g., "KNN", "MBK")
        model_class: Model class to instantiate
        model_params: Parameters for the model
        dataset_source: Source type of the dataset
        dataset_name: Dataset identifier
        dataset_label: Human-readable dataset name
        seed: Random seed for reproducibility
        paradigm: "supervised" or "unsupervised"
        impact_mode: If True, run full impact analysis with isolated perturbations

    Returns:
        dict: Experiment results
    """
    # Update random_state in params if the model supports it
    params = model_params.copy()
    if "random_state" in params:
        params["random_state"] = seed

    # Set n_clusters/n_components for unsupervised algorithms
    if paradigm == "unsupervised" and dataset_label in DATASET_N_CLUSTERS:
        n_clusters = DATASET_N_CLUSTERS[dataset_label]
        if "n_clusters" in params:
            params["n_clusters"] = n_clusters
        if "n_components" in params:
            params["n_components"] = n_clusters

    mode_str = "(IMPACT ANALYSIS)" if impact_mode else ""
    logger.info(
        f"Running [{paradigm.upper()}]: {algorithm_name} on {dataset_label} "
        f"with seed={seed} {mode_str}"
    )

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
                "paradigm": paradigm,
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
                "paradigm": paradigm,
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
            "paradigm": paradigm,
            "dataset": dataset_label,
            "seed": seed,
            "error": str(e),
        }


def run_case_study_2(
    output_dir: str = "results/case2",
    algorithms: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    paradigms: Optional[List[str]] = None,
    dry_run: bool = False,
    impact_mode: bool = False,
) -> dict:
    """Run Case Study 2: Paradigm Comparison.

    Args:
        output_dir: Directory to save results
        algorithms: List of algorithm names to run (None = all)
        datasets: List of dataset names to run (None = all)
        seeds: List of seeds to use (None = all 5)
        paradigms: List of paradigms to run ("supervised", "unsupervised", or both)
        dry_run: If True, only print what would be run
        impact_mode: If True, run P2 impact analysis with isolated perturbations

    Returns:
        dict: Summary of all experiment results
    """
    # Default paradigms
    paradigm_list = paradigms or ["supervised", "unsupervised"]

    # Build experiment configurations
    experiments_to_run = []

    if "supervised" in paradigm_list:
        algo_list = algorithms or list(SUPERVISED_ALGORITHMS.keys())
        algo_configs = {k: v for k, v in SUPERVISED_ALGORITHMS.items() if k in algo_list}

        dataset_list = datasets or [d[2] for d in SUPERVISED_DATASETS]
        dataset_configs = [d for d in SUPERVISED_DATASETS if d[2] in dataset_list]

        for algo_name, (model_class, model_params) in algo_configs.items():
            for dataset_source, dataset_name, dataset_label in dataset_configs:
                experiments_to_run.append(
                    {
                        "algorithm": algo_name,
                        "model_class": model_class,
                        "model_params": model_params,
                        "dataset_source": dataset_source,
                        "dataset_name": dataset_name,
                        "dataset_label": dataset_label,
                        "paradigm": "supervised",
                    }
                )

    if "unsupervised" in paradigm_list:
        algo_list = algorithms or list(UNSUPERVISED_ALGORITHMS.keys())
        algo_configs = {k: v for k, v in UNSUPERVISED_ALGORITHMS.items() if k in algo_list}

        dataset_list = datasets or [d[2] for d in UNSUPERVISED_DATASETS]
        dataset_configs = [d for d in UNSUPERVISED_DATASETS if d[2] in dataset_list]

        for algo_name, (model_class, model_params) in algo_configs.items():
            for dataset_source, dataset_name, dataset_label in dataset_configs:
                experiments_to_run.append(
                    {
                        "algorithm": algo_name,
                        "model_class": model_class,
                        "model_params": model_params,
                        "dataset_source": dataset_source,
                        "dataset_name": dataset_name,
                        "dataset_label": dataset_label,
                        "paradigm": "unsupervised",
                    }
                )

    # Filter seeds
    seed_list = seeds or SEEDS

    # Calculate total experiments
    total_experiments = len(experiments_to_run) * len(seed_list)

    # Count by paradigm
    supervised_count = sum(1 for e in experiments_to_run if e["paradigm"] == "supervised")
    unsupervised_count = sum(1 for e in experiments_to_run if e["paradigm"] == "unsupervised")

    logger.info("=" * 70)
    logger.info("CASE STUDY 2: PARADIGM COMPARISON - SUPERVISED VS UNSUPERVISED")
    if impact_mode:
        logger.info("MODE: IMPACT ANALYSIS (P2) - Isolated perturbation effects")
    logger.info("=" * 70)
    logger.info(f"Paradigms: {paradigm_list}")
    logger.info(
        f"Supervised experiments: {supervised_count} configs × {len(seed_list)} seeds = {supervised_count * len(seed_list)}"
    )
    logger.info(
        f"Unsupervised experiments: {unsupervised_count} configs × {len(seed_list)} seeds = {unsupervised_count * len(seed_list)}"
    )
    logger.info(f"Seeds: {seed_list}")
    logger.info(f"Total experiment pairs: {total_experiments}")
    logger.info(f"Total scenario runs: {total_experiments * 2} (baseline + SIP each)")
    logger.info("=" * 70)

    if dry_run:
        logger.info("DRY RUN - No experiments will be executed")
        logger.info("\nExperiments that would be run:")
        for exp in experiments_to_run:
            logger.info(
                f"  [{exp['paradigm'].upper()}] {exp['algorithm']} on {exp['dataset_label']}"
            )
        return {"dry_run": True, "total_experiments": total_experiments}

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    artifact_snapshot = snapshot_default_artifacts()

    # Track results
    all_results = []
    summary = {
        "study": "Case Study 2 - Paradigm Comparison (Supervised vs Unsupervised)",
        "timestamp": get_timestamp(),
        "configuration": {
            "paradigms": paradigm_list,
            "supervised_algorithms": [
                e["algorithm"] for e in experiments_to_run if e["paradigm"] == "supervised"
            ],
            "unsupervised_algorithms": [
                e["algorithm"] for e in experiments_to_run if e["paradigm"] == "unsupervised"
            ],
            "supervised_datasets": list(
                set(e["dataset_label"] for e in experiments_to_run if e["paradigm"] == "supervised")
            ),
            "unsupervised_datasets": list(
                set(
                    e["dataset_label"]
                    for e in experiments_to_run
                    if e["paradigm"] == "unsupervised"
                )
            ),
            "seeds": seed_list,
            "total_experiments": total_experiments,
        },
        "results_by_paradigm": {"supervised": [], "unsupervised": []},
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

    for exp in experiments_to_run:
        algo_name = exp["algorithm"]
        if algo_name not in summary["results_by_algorithm"]:
            summary["results_by_algorithm"][algo_name] = []

        dataset_label = exp["dataset_label"]
        if dataset_label not in summary["results_by_dataset"]:
            summary["results_by_dataset"][dataset_label] = []

        for seed in seed_list:
            experiment_count += 1
            logger.info(f"\n[{experiment_count}/{total_experiments}]")

            result = run_single_experiment(
                algorithm_name=algo_name,
                model_class=exp["model_class"],
                model_params=exp["model_params"],
                dataset_source=exp["dataset_source"],
                dataset_name=exp["dataset_name"],
                dataset_label=dataset_label,
                seed=seed,
                paradigm=exp["paradigm"],
                impact_mode=impact_mode,
            )

            all_results.append(result)
            summary["results_by_paradigm"][exp["paradigm"]].append(result)
            summary["results_by_algorithm"][algo_name].append(result)
            summary["results_by_dataset"][dataset_label].append(result)

            if result["status"] == "success":
                summary["overall_stats"]["successful"] += 1
            else:
                summary["overall_stats"]["failed"] += 1

    summary["overall_stats"]["total_time_seconds"] = time.time() - start_time

    # Save all results
    results_file = os.path.join(output_dir, f"case2_all_results_{get_timestamp()}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"All results saved to: {results_file}")

    # Save summary
    summary_file = os.path.join(output_dir, f"case2_summary_{get_timestamp()}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to: {summary_file}")

    moved_artifacts = relocate_new_default_artifacts(
        snapshot=artifact_snapshot, output_dir=output_dir
    )
    if moved_artifacts:
        logger.info("Relocated %d auxiliary artifacts to %s", len(moved_artifacts), output_dir)

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("CASE STUDY 2 COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Successful: {summary['overall_stats']['successful']}")
    logger.info(f"Failed: {summary['overall_stats']['failed']}")
    logger.info(
        f"  - Supervised: {len([r for r in all_results if r.get('paradigm') == 'supervised' and r['status'] == 'success'])} success"
    )
    logger.info(
        f"  - Unsupervised: {len([r for r in all_results if r.get('paradigm') == 'unsupervised' and r['status'] == 'success'])} success"
    )
    logger.info(f"Total time: {summary['overall_stats']['total_time_seconds']:.2f} seconds")
    logger.info("=" * 70)

    return summary


def main():
    """Main entry point for Case Study 2."""
    all_algorithms = list(SUPERVISED_ALGORITHMS.keys()) + list(UNSUPERVISED_ALGORITHMS.keys())
    all_datasets = [d[2] for d in SUPERVISED_DATASETS] + [d[2] for d in UNSUPERVISED_DATASETS]

    parser = argparse.ArgumentParser(
        description="Case Study 2: Paradigm Comparison - Supervised vs Unsupervised",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python case2_sip_paradigmas_super_nao_super.py

  # Run only supervised paradigm
  python case2_sip_paradigmas_super_nao_super.py --paradigms supervised

  # Run only unsupervised paradigm
  python case2_sip_paradigmas_super_nao_super.py --paradigms unsupervised

  # Run specific algorithms
  python case2_sip_paradigmas_super_nao_super.py --algorithms KNN SVM MBK GMM

  # Run with specific datasets
  python case2_sip_paradigmas_super_nao_super.py --datasets Wine Digits "Make Moons"

  # Dry run to see what would be executed
  python case2_sip_paradigmas_super_nao_super.py --dry-run

  # Run single experiment for quick test
  python case2_sip_paradigmas_super_nao_super.py --algorithms KNN --datasets Wine --seeds 42 --paradigms supervised
        """,
    )

    parser.add_argument(
        "--paradigms",
        nargs="+",
        choices=["supervised", "unsupervised"],
        help="Paradigms to run (default: both)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=all_algorithms,
        help="Algorithms to run (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=all_datasets,
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
        default="results/case2",
        help="Output directory for results (default: results/case2)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/case2",
        help="Directory for case2 execution logs (default: logs/case2)",
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
    pre_run_snapshot = snapshot_default_artifacts()
    log_file = configure_case_file_logger(logger, args.log_dir, "case2")
    logger.info("Case2 log file: %s", log_file)

    run_case_study_2(
        output_dir=args.output_dir,
        algorithms=args.algorithms,
        datasets=args.datasets,
        seeds=args.seeds,
        paradigms=args.paradigms,
        dry_run=args.dry_run,
        impact_mode=args.impact_analysis,
    )
    moved_after_main = relocate_new_default_artifacts(
        snapshot=pre_run_snapshot, output_dir=args.output_dir
    )
    if moved_after_main:
        logger.info(
            "Main relocation moved %d additional artifacts to %s",
            len(moved_after_main),
            args.output_dir,
        )


if __name__ == "__main__":
    main()
