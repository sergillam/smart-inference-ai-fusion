"""Experiment script for AgglomerativeClustering on the Digits dataset."""

from __future__ import annotations

import numpy as np

from smart_inference_ai_fusion.core.experiment import Experiment
from smart_inference_ai_fusion.datasets.factory import DatasetFactory
from smart_inference_ai_fusion.inference.engine.param_runner import ParameterInferenceEngine
from smart_inference_ai_fusion.inference.pipeline.inference_pipeline import InferencePipeline
from smart_inference_ai_fusion.models.agglomerative_clustering_model import (
    AgglomerativeClusteringModel,
)
from smart_inference_ai_fusion.utils.preprocessing import filter_sklearn_params
from smart_inference_ai_fusion.utils.report import (
    ReportMode,
    generate_experiment_filename,
    report_data,
)
from smart_inference_ai_fusion.utils.types import (
    DataNoiseConfig,
    DatasetSourceType,
    LabelNoiseConfig,
    ParameterNoiseConfig,
    SklearnDatasetName,
)

# pylint: disable=duplicate-code

# ---- enums válidos (scikit-learn atuais) ----
_VALID_LINKAGE = {"ward", "complete", "average", "single"}
# 'metric' depende do linkage (ward -> 'euclidean' obrigatório)
_VALID_METRIC = {
    "euclidean",
    "l1",
    "l2",
    "manhattan",
    "cosine",
    "precomputed",
}


def _coerce_numeric_labels(y):
    """Converts numeric strings to integers when possible (safe round-tripping)."""
    y_arr = np.asarray(y)
    if np.issubdtype(y_arr.dtype, np.integer):
        return y_arr
    try:
        y_int = y_arr.astype(int)
        if np.all(y_int.astype(str) == y_arr):
            return y_int
    except (ValueError, TypeError):
        return y_arr
    return y_arr


# ---- sanitização de parâmetros (decomposta p/ evitar McCabe/branches) ----
def _sanitize_enums(p: dict) -> None:
    linkage = p.get("linkage")
    if linkage not in _VALID_LINKAGE:
        p["linkage"] = "ward"

    metric = p.get("metric")
    if p["linkage"] == "ward":
        p["metric"] = "euclidean"
    else:
        if metric not in _VALID_METRIC:
            p["metric"] = "euclidean"


def _sanitize_ints(p: dict, y_train) -> None:
    # travar n_clusters ao nº de classes quando possível (mínimo 2)
    try:
        n_classes = int(len(np.unique(y_train)))
    except (ValueError, TypeError):
        n_classes = int(p.get("n_clusters", 10) or 10)
    p["n_clusters"] = max(2, n_classes)

    # distance_threshold é mutuamente exclusivo com n_clusters
    p["distance_threshold"] = None


def _sanitize_flags(p: dict) -> None:
    # compute_full_tree pode ser "auto" | bool
    if "compute_full_tree" not in p:
        p["compute_full_tree"] = "auto"


def _sanitize_agglom_params(params: dict, y_train) -> dict:
    """Sanitiza parâmetros do AgglomerativeClustering de forma robusta."""
    p = dict(params) if params is not None else {}
    _sanitize_enums(p)
    _sanitize_ints(p, y_train)
    _sanitize_flags(p)
    return p


# ---- Experimentos ----
def run_agglomerative_without_inference() -> None:
    """Baseline (sem inferência/perturbações)."""
    model_class = AgglomerativeClusteringModel
    dataset_name = SklearnDatasetName.DIGITS
    name_output = generate_experiment_filename(model_class, dataset_name)

    report_data("=== AgglomerativeClustering WITHOUT INFERENCE ===", mode=ReportMode.PRINT)

    base_params = {
        "n_clusters": 10,
        "linkage": "ward",  # ward = métrica euclidiana obrigatória
        "metric": "euclidean",
        "compute_full_tree": "auto",
        "distance_threshold": None,
    }

    filtered_params = filter_sklearn_params(base_params, AgglomerativeClusteringModel)
    model = model_class(**filtered_params)

    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=dataset_name)
    x_train, x_test, y_train, y_test = dataset.load_data()

    experiment = Experiment(model, dataset)
    metrics = experiment.run(x_train, x_test, y_train, y_test)

    report_data("Evaluation metrics (no inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.JSON_RESULT, name_output=f"{name_output}-no-inference")


def run_agglomerative_with_inference() -> None:
    """Executa Agglomerative com inferência (dados, labels e parâmetros)."""
    model_class = AgglomerativeClusteringModel
    dataset_name = SklearnDatasetName.DIGITS
    name_output = generate_experiment_filename(model_class, dataset_name)

    report_data(
        "=== AgglomerativeClustering WITH INFERENCE (data + param + label) ===",
        mode=ReportMode.PRINT,
    )

    base_params = {
        "n_clusters": 10,
        "linkage": "ward",
        "metric": "euclidean",
        "compute_full_tree": "auto",
        "distance_threshold": None,
    }

    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=dataset_name)

    data_noise_config = DataNoiseConfig(
        noise_level=0.2,
        truncate_decimals=1,
        quantize_bins=5,
        cast_to_int=False,
        shuffle_fraction=0.1,
        scale_range=(0.8, 1.2),
        zero_out_fraction=0.05,
        insert_nan_fraction=0.05,
        outlier_fraction=0.05,
        add_dummy_features=2,
        duplicate_features=2,
        feature_selective_noise=(0.3, [0, 2]),
        remove_features=[1, 3],
        feature_swap=[0, 2],
        conditional_noise=(0, 5.0, 0.2),
        random_missing_block_fraction=0.1,
        distribution_shift_fraction=0.1,
        cluster_swap_fraction=0.1,
        group_outlier_cluster_fraction=0.1,
        temporal_drift_std=0.5,
    )
    label_noise_config = LabelNoiseConfig(
        label_noise_fraction=0.05,
        flip_near_border_fraction=0.05,
        confusion_matrix_noise_level=0.05,
        partial_label_fraction=0.05,
        swap_within_class_fraction=0.05,
    )
    param_noise_config = ParameterNoiseConfig(
        integer_noise=True,
        boolean_flip=False,
        string_mutator=False,
        semantic_mutation=False,
        scale_hyper=True,
        cross_dependency=False,
        random_from_space=False,
        bounded_numeric=True,
        type_cast_perturbation=False,
        enum_boundary_shift=False,
    )

    x_train, x_test, y_train, y_test = dataset.load_data()

    pipeline = InferencePipeline(
        data_noise_config=data_noise_config,
        label_noise_config=label_noise_config,
        X_train=x_train,
    )

    # Perturba params e sanitiza
    param_engine = ParameterInferenceEngine(config=param_noise_config)
    perturbed = param_engine.apply(base_params)
    # Garantir coerência ward/euclidean e n_clusters
    perturbed = _sanitize_agglom_params(perturbed, y_train)

    report_data(f"Perturbed parameters: {perturbed}", mode=ReportMode.PRINT)
    report_data(
        param_engine.export_log(),
        mode=ReportMode.JSON_LOG,
        name_output=f"{name_output}-param-perturb",
    )

    filtered_params = filter_sklearn_params(perturbed, AgglomerativeClusteringModel)
    model = AgglomerativeClusteringModel(**filtered_params)

    # Inferência nos dados
    x_train_inf, x_test_inf = pipeline.apply_data_inference(x_train, x_test)
    model.fit(x_train_inf, y_train)

    # Inferência nos rótulos + coerção para inteiros quando possível
    y_train_inf, y_test_inf = pipeline.apply_label_inference(
        y_train, y_test, model=model, X_train=x_train_inf, X_test=x_test_inf
    )
    y_train_inf = _coerce_numeric_labels(y_train_inf)
    y_test_inf = _coerce_numeric_labels(y_test_inf)

    experiment = Experiment(model, dataset)
    metrics = experiment.run(x_train_inf, x_test_inf, y_train_inf, y_test_inf)

    report_data("Evaluation metrics (with inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.JSON_RESULT, name_output=f"{name_output}-with-inference")


def run() -> None:
    """Roda os dois cenários (sem/como inferência)."""
    run_agglomerative_without_inference()
    run_agglomerative_with_inference()


if __name__ == "__main__":
    run()
