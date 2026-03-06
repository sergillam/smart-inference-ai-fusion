#!/usr/bin/env python
"""Case Study 4: SIP-Q quantization experiment runner."""
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.results_io import load_json_records
from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.experiments.quantization_experiment import QuantizationExperiment
from smart_inference_ai_fusion.models.agglomerative_clustering_model import (
    AgglomerativeClusteringModel,
)
from smart_inference_ai_fusion.models.gaussian_mixture_model import GaussianMixtureModel
from smart_inference_ai_fusion.models.knn_model import KNNModel
from smart_inference_ai_fusion.models.minibatch_kmeans_model import MiniBatchKMeansModel
from smart_inference_ai_fusion.models.mlp_model import MLPModel
from smart_inference_ai_fusion.models.tree_model import DecisionTreeModel
from smart_inference_ai_fusion.quantization.core import QuantizationConfig, QuantizationResult
from smart_inference_ai_fusion.utils.types import DatasetSourceType, SklearnDatasetName

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]

SUPERVISED_DATASETS = {
    "Wine": (DatasetSourceType.SKLEARN, SklearnDatasetName.WINE),
    "Digits": (DatasetSourceType.SKLEARN, SklearnDatasetName.DIGITS),
}

UNSUPERVISED_DATASETS = {
    "MakeBlobs": (DatasetSourceType.SKLEARN, SklearnDatasetName.MAKE_BLOBS),
    "MakeMoons": (DatasetSourceType.SKLEARN, SklearnDatasetName.MAKE_MOONS),
}

SUPERVISED_ALGOS: dict[str, tuple[type[BaseModel], dict[str, Any]]] = {
    "KNN": (KNNModel, {"n_neighbors": 5}),
    "DT": (DecisionTreeModel, {"max_depth": 10, "random_state": None}),
    "MLP": (MLPModel, {"hidden_layer_sizes": (100,), "max_iter": 500, "random_state": None}),
}

UNSUPERVISED_ALGOS: dict[str, tuple[type[BaseModel], dict[str, Any]]] = {
    "MBK": (MiniBatchKMeansModel, {"n_clusters": 3, "random_state": None}),
    "GMM": (GaussianMixtureModel, {"n_components": 3, "random_state": None}),
    "AC": (AgglomerativeClusteringModel, {"n_clusters": 3}),
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case Study 4: SIP-Q Quantization")
    parser.add_argument(
        "--datasets", nargs="+", default=["Wine", "Digits", "MakeBlobs", "MakeMoons"]
    )
    parser.add_argument("--algorithms", nargs="+", default=["KNN", "DT", "MLP", "MBK", "GMM", "AC"])
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--method", default="uniform")
    parser.add_argument("--dtype-profile", default="integer", choices=["integer", "float16"])
    parser.add_argument(
        "--output-dir",
        default="results/case4",
        help="Output directory for results (default: results/case4)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/case4",
        help="Directory for case4 execution logs (default: logs/case4)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print run plan without executing")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip execution_ids already present in output JSON files.",
    )
    return parser.parse_args()


def _load_existing_execution_ids(output_dir: Path) -> set[str]:
    execution_ids: set[str] = set()
    for pattern in ("case4_all_results_*.json", "case4_results_*.json"):
        for record in load_json_records(output_dir, pattern):
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict):
                exec_id = metadata.get("execution_id")
                if isinstance(exec_id, str):
                    execution_ids.add(exec_id)
    return execution_ids


def _ensure_float16_constraints(dtype_profile: str, bits: list[int]) -> None:
    if dtype_profile == "float16" and any(bit != 16 for bit in bits):
        raise ValueError("For dtype_profile='float16', --bits must contain only 16.")


def _update_result_metadata(
    result: QuantizationResult,
    *,
    dataset_label: str,
    algorithm_key: str,
    seed: int,
) -> None:
    result.metadata.update(
        {
            "dataset": dataset_label,
            "algorithm": algorithm_key,
            "seed": seed,
        }
    )


def _run_one_configuration(
    *,
    experiment: QuantizationExperiment,
    dataset_label: str,
    source: DatasetSourceType,
    dataset_name: SklearnDatasetName,
    algo_key: str,
    model_class: type[BaseModel],
    base_params: dict[str, Any],
    seed: int,
    skip_execution_ids: set[str],
    supervised: bool,
) -> list[QuantizationResult]:
    model_params = dict(base_params)
    if "random_state" in model_params:
        model_params["random_state"] = seed

    if supervised:
        results = experiment.run_supervised(
            source,
            dataset_name,
            model_class,
            model_params,
            seed=seed,
            skip_execution_ids=skip_execution_ids,
        )
    else:
        results = experiment.run_unsupervised(
            source,
            dataset_name,
            model_class,
            model_params,
            seed=seed,
            skip_execution_ids=skip_execution_ids,
        )

    for result in results:
        _update_result_metadata(
            result, dataset_label=dataset_label, algorithm_key=algo_key, seed=seed
        )
        exec_id = result.metadata.get("execution_id")
        if isinstance(exec_id, str):
            skip_execution_ids.add(exec_id)
    return results


def _run_dataset_group(
    *,
    experiment: QuantizationExperiment,
    datasets: dict[str, tuple[DatasetSourceType, SklearnDatasetName]],
    algorithms: dict[str, tuple[type[BaseModel], dict[str, Any]]],
    selected_datasets: set[str],
    selected_algorithms: set[str],
    seed: int,
    skip_execution_ids: set[str],
    supervised: bool,
) -> list[QuantizationResult]:
    results: list[QuantizationResult] = []
    for dataset_label, (source, dataset_name) in datasets.items():
        if dataset_label not in selected_datasets:
            continue
        for algo_key, (model_class, base_params) in algorithms.items():
            if algo_key not in selected_algorithms:
                continue
            results.extend(
                _run_one_configuration(
                    experiment=experiment,
                    dataset_label=dataset_label,
                    source=source,
                    dataset_name=dataset_name,
                    algo_key=algo_key,
                    model_class=model_class,
                    base_params=base_params,
                    seed=seed,
                    skip_execution_ids=skip_execution_ids,
                    supervised=supervised,
                )
            )
    return results


def _configure_file_logger(log_dir: str, stamp: str) -> Path:
    # Keep a single active file handler even when main() runs multiple times in-process.
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
    os.makedirs(log_dir, exist_ok=True)
    log_file = Path(log_dir) / f"case4_{stamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return log_file


def run_case_study_4(
    output_dir: str = "results/case4",
    *,
    datasets: list[str] | None = None,
    algorithms: list[str] | None = None,
    bits: list[int] | None = None,
    seeds: list[int] | None = None,
    method: str = "uniform",
    dtype_profile: str = "integer",
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run Case Study 4 and persist full results + summary."""
    selected_datasets = set(datasets or ["Wine", "Digits", "MakeBlobs", "MakeMoons"])
    selected_algorithms = set(algorithms or ["KNN", "DT", "MLP", "MBK", "GMM", "AC"])
    selected_bits = bits or [8, 16, 32]
    seed_list = seeds or SEEDS

    _ensure_float16_constraints(dtype_profile, selected_bits)

    supervised_total = sum(1 for d in selected_datasets if d in SUPERVISED_DATASETS) * sum(
        1 for a in selected_algorithms if a in SUPERVISED_ALGOS
    )
    unsupervised_total = sum(1 for d in selected_datasets if d in UNSUPERVISED_DATASETS) * sum(
        1 for a in selected_algorithms if a in UNSUPERVISED_ALGOS
    )
    total_configurations = supervised_total + unsupervised_total
    valid_bits = (
        sorted({bit for bit in selected_bits if bit == 16})
        if dtype_profile == "float16"
        else sorted(set(selected_bits))
    )
    modes_per_bit = 3  # data_only, model_only, hybrid
    total_runs = total_configurations * len(seed_list) * len(valid_bits) * modes_per_bit

    logger.info("=" * 70)
    logger.info("CASE STUDY 4: SIP-Q QUANTIZATION EVALUATION")
    logger.info("=" * 70)
    logger.info("Datasets: %s", sorted(selected_datasets))
    logger.info("Algorithms: %s", sorted(selected_algorithms))
    logger.info("Bits: %s | dtype_profile=%s | method=%s", selected_bits, dtype_profile, method)
    logger.info("Seeds: %s", seed_list)
    logger.info("Total configurations: %s", total_configurations)
    logger.info("Total runs: %s", total_runs)
    logger.info("=" * 70)

    if dry_run:
        logger.info("DRY RUN - No experiments will be executed")
        return {
            "dry_run": True,
            "total_configurations": total_configurations,
            "total_runs": total_runs,
        }

    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir)
    skip_execution_ids = _load_existing_execution_ids(output_path) if resume else set()

    all_results: list[QuantizationResult] = []
    summary: dict[str, Any] = {
        "study": "Case Study 4 - SIP-Q Quantization Evaluation",
        "timestamp": _timestamp(),
        "configuration": {
            "datasets": sorted(selected_datasets),
            "algorithms": sorted(selected_algorithms),
            "bits": selected_bits,
            "seeds": seed_list,
            "method": method,
            "dtype_profile": dtype_profile,
            "resume": resume,
            "total_configurations": total_configurations,
            "total_runs": total_runs,
        },
        "results_by_dataset": {},
        "results_by_algorithm": {},
        "overall_stats": {
            "records_generated": 0,
            "total_time_seconds": 0.0,
        },
    }

    start_time = time.time()

    for seed in seed_list:
        config = QuantizationConfig(
            data_bits=tuple(selected_bits),
            model_bits=tuple(selected_bits),
            dtype_profile=dtype_profile,
            method=method,
            enable_hybrid=True,
            random_seed=seed,
        )
        experiment = QuantizationExperiment(config)
        all_results.extend(
            _run_dataset_group(
                experiment=experiment,
                datasets=SUPERVISED_DATASETS,
                algorithms=SUPERVISED_ALGOS,
                selected_datasets=selected_datasets,
                selected_algorithms=selected_algorithms,
                seed=seed,
                skip_execution_ids=skip_execution_ids,
                supervised=True,
            )
        )
        all_results.extend(
            _run_dataset_group(
                experiment=experiment,
                datasets=UNSUPERVISED_DATASETS,
                algorithms=UNSUPERVISED_ALGOS,
                selected_datasets=selected_datasets,
                selected_algorithms=selected_algorithms,
                seed=seed,
                skip_execution_ids=skip_execution_ids,
                supervised=False,
            )
        )

    for result in all_results:
        dataset = result.metadata.get("dataset", result.dataset_name)
        algorithm = result.metadata.get("algorithm", result.algorithm_name)
        summary["results_by_dataset"].setdefault(dataset, 0)
        summary["results_by_dataset"][dataset] += 1
        summary["results_by_algorithm"].setdefault(algorithm, 0)
        summary["results_by_algorithm"][algorithm] += 1

    summary["overall_stats"]["records_generated"] = len(all_results)
    summary["overall_stats"]["total_time_seconds"] = float(time.time() - start_time)

    stamp = _timestamp()
    results_file = output_path / f"case4_all_results_{stamp}.json"
    summary_file = output_path / f"case4_summary_{stamp}.json"

    with open(results_file, "w", encoding="utf-8") as handle:
        json.dump([result.model_dump() for result in all_results], handle, indent=2)
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("All results saved to: %s", results_file)
    logger.info("Summary saved to: %s", summary_file)
    logger.info("Total records generated: %s", len(all_results))
    logger.info("Total time: %.2f seconds", summary["overall_stats"]["total_time_seconds"])
    logger.info("=" * 70)
    return summary


def main() -> None:
    """Run Case 4 experiments from CLI."""
    args = _parse_args()
    stamp = _timestamp()
    log_file = _configure_file_logger(args.log_dir, stamp)
    logger.info("Case4 log file: %s", log_file)
    run_case_study_4(
        output_dir=args.output_dir,
        datasets=args.datasets,
        algorithms=args.algorithms,
        bits=args.bits,
        seeds=args.seeds,
        method=args.method,
        dtype_profile=args.dtype_profile,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
