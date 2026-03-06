#!/usr/bin/env python
"""Case Study 4: SIP-Q quantization experiment runner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIP-Q Case Study 4")
    parser.add_argument(
        "--datasets", nargs="+", default=["Wine", "Digits", "MakeBlobs", "MakeMoons"]
    )
    parser.add_argument("--algorithms", nargs="+", default=["KNN", "DT", "MLP", "MBK", "GMM", "AC"])
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--method", default="uniform")
    parser.add_argument("--dtype-profile", default="integer", choices=["integer", "float16"])
    parser.add_argument("--output", default="results/case4")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip execution_ids already present in output JSON files.",
    )
    return parser.parse_args()


def _load_existing_execution_ids(output_dir: Path) -> set[str]:
    execution_ids: set[str] = set()
    for record in load_json_records(output_dir, "case4_results_*.json"):
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


def main() -> None:
    """Run Case 4 experiments and save consolidated JSON results."""
    args = _parse_args()
    _ensure_float16_constraints(args.dtype_profile, args.bits)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_execution_ids = _load_existing_execution_ids(output_dir) if args.resume else set()
    selected_datasets = set(args.datasets)
    selected_algorithms = set(args.algorithms)
    all_results: list[QuantizationResult] = []

    for seed in args.seeds:
        config = QuantizationConfig(
            data_bits=tuple(args.bits),
            model_bits=tuple(args.bits),
            dtype_profile=args.dtype_profile,
            method=args.method,
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"case4_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump([result.model_dump() for result in all_results], handle, indent=2)

    print(f"Saved {len(all_results)} records to {output_file}")


if __name__ == "__main__":
    main()
