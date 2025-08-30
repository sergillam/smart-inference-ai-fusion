"""Experiment script for FastICA(+KMeans) on the Digits dataset."""

from __future__ import annotations

import numpy as np

from smart_inference_ai_fusion.core.experiment import Experiment
from smart_inference_ai_fusion.datasets.factory import DatasetFactory
from smart_inference_ai_fusion.inference.engine.param_runner import ParameterInferenceEngine
from smart_inference_ai_fusion.inference.pipeline.inference_pipeline import InferencePipeline
from smart_inference_ai_fusion.models.fastica_model import FastICAModel
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


def _coerce_numeric_labels(y):
    """Converte rótulos para inteiros quando possível (sem perdas) e garante vetor 1D.

    - Se dtype já é inteiro, apenas retorna y.ravel().
    - Se são strings que representam inteiros, converte para int (com checagem round-trip).
    - Caso contrário, mantém dtype original, mas ravel() para evitar erros de forma.
    """
    arr = np.asarray(y)
    arr = arr.ravel()
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    # tenta converter strings que “são” inteiros
    try:
        as_int = arr.astype(int)
        if np.all(as_int.astype(str) == arr):
            return as_int
    except (TypeError, ValueError):
        pass
    return arr


def _lock_kmeans_clusters_to_dataset(params: dict, y_train) -> dict:
    """Garante kmeans_n_clusters == nº de classes observadas (>= 2)."""
    p = dict(params)
    try:
        n_classes = int(len(np.unique(np.asarray(y_train).ravel())))
        p["kmeans_n_clusters"] = max(2, n_classes)
    except (TypeError, ValueError):
        p["kmeans_n_clusters"] = max(2, int(p.get("kmeans_n_clusters", 10)))
    return p


# HELPER FUNCTIONS


def _sanitize_int_param(p: dict, key: str, default: int, min_val: int = 1) -> None:
    """Sanitizes a single integer parameter in-place."""
    try:
        val = int(p.get(key, default))
        p[key] = val if val >= min_val else default
    except (ValueError, TypeError):
        p[key] = default


def _sanitize_ica_specifics(p: dict) -> None:
    """Handles unique validation logic for FastICA parameters in-place."""
    # Sanitize enum-like string parameters
    if p.get("algorithm") not in {"parallel", "deflation"}:
        p["algorithm"] = "parallel"
    if p.get("whiten") not in {True, False, "unit-variance"}:
        p["whiten"] = "unit-variance"

    # Sanitize n_components, which can be None
    try:
        n_comp = p.get("n_components")
        if n_comp is not None:
            p["n_components"] = int(n_comp) if int(n_comp) > 1 else None
    except (ValueError, TypeError):
        p["n_components"] = None


# MAIN REFACTORED FUNCTION


def _sanitize_fastica_params(params: dict) -> dict:
    """Cleans and validates a dictionary of FastICA and KMeans parameters.

    This function delegates validation to smaller helpers to reduce complexity.
    """
    p = dict(params)

    # 1. Handle ICA-specific parameters
    _sanitize_ica_specifics(p)
    _sanitize_int_param(p, "max_iter", default=400, min_val=1)

    # 2. Handle KMeans-specific parameters
    _sanitize_int_param(p, "kmeans_n_clusters", default=10, min_val=2)
    _sanitize_int_param(p, "kmeans_n_init", default=10, min_val=1)

    # 3. Handle shared parameters like random_state
    try:
        rs = p.get("random_state")
        p["random_state"] = int(rs) if rs is not None else 42
    except (ValueError, TypeError):
        p["random_state"] = 42

    return p


# pylint: disable=duplicate-code
def run_fastica_without_inference() -> None:
    """Run the FastICA(+KMeans) baseline (no inference or perturbations)."""
    model_class = FastICAModel
    dataset_name = SklearnDatasetName.DIGITS
    name_output = generate_experiment_filename(model_class, dataset_name)

    report_data("=== FastICA(+KMeans) WITHOUT INFERENCE ===", mode=ReportMode.PRINT)

    # baseline razoável para digits
    base_params = {
        "n_components": 40,  # <= 64 (n_features) — bom compromisso estabilidade/desempenho
        "algorithm": "parallel",
        "whiten": "unit-variance",
        "random_state": 42,
        "max_iter": 400,
        "tol": 1e-4,
        "kmeans_n_clusters": 10,
        "kmeans_n_init": 10,
        "kmeans_random_state": 42,
    }

    filtered_params = filter_sklearn_params(base_params, FastICAModel)
    model = model_class(**filtered_params)

    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=dataset_name)
    X_train, X_test, y_train, y_test = dataset.load_data()

    # Coerção defensiva de rótulos (evita dtype/shape problemáticos nas métricas)
    y_train = _coerce_numeric_labels(y_train)
    y_test = _coerce_numeric_labels(y_test)

    experiment = Experiment(model, dataset)
    metrics = experiment.run(X_train, X_test, y_train, y_test)

    report_data("Evaluation metrics (no inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.JSON_RESULT, name_output=f"{name_output}-no-inference")


def run_fastica_with_inference() -> None:
    """Run the FastICA(+KMeans) model with data/param/label inference."""
    model_class = FastICAModel
    dataset_name = SklearnDatasetName.DIGITS
    name_output = generate_experiment_filename(model_class, dataset_name)

    report_data(
        "=== FastICA(+KMeans) WITH INFERENCE (data + param + label) ===",
        mode=ReportMode.PRINT,
    )

    # ponto de partida
    base_params = {
        "n_components": 40,
        "algorithm": "parallel",
        "whiten": "unit-variance",
        "random_state": 42,
        "max_iter": 400,
        "tol": 1e-4,
        "kmeans_n_clusters": 10,
        "kmeans_n_init": 10,
        "kmeans_random_state": 42,
    }

    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=dataset_name)

    # ruídos / inferências
    data_noise_config = DataNoiseConfig(
        noise_level=0.15,
        truncate_decimals=1,
        quantize_bins=5,
        cast_to_int=False,
        shuffle_fraction=0.1,
        scale_range=(0.9, 1.1),
        zero_out_fraction=0.03,
        insert_nan_fraction=0.02,
        outlier_fraction=0.03,
        add_dummy_features=1,
        duplicate_features=1,
        feature_selective_noise=(0.25, [0, 2]),
        remove_features=[],
        feature_swap=None,
        conditional_noise=None,
        random_missing_block_fraction=0.08,
        distribution_shift_fraction=0.08,
        cluster_swap_fraction=0.0,
        group_outlier_cluster_fraction=0.0,
        temporal_drift_std=0.3,
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

    # perturba e saneia
    param_engine = ParameterInferenceEngine(config=param_noise_config)
    perturbed_params = param_engine.apply(base_params)
    perturbed_params = _sanitize_fastica_params(perturbed_params)
    perturbed_params = _lock_kmeans_clusters_to_dataset(perturbed_params, y_train)

    report_data(f"Perturbed parameters: {perturbed_params}", mode=ReportMode.PRINT)
    report_data(
        param_engine.export_log(),
        mode=ReportMode.JSON_LOG,
        name_output=f"{name_output}-param-perturb",
    )

    filtered_params = filter_sklearn_params(perturbed_params, FastICAModel)
    model = FastICAModel(**filtered_params)

    # inferência de dados e labels
    x_train_inferred, x_test_inferred = pipeline.apply_data_inference(x_train, x_test)
    model.fit(x_train_inferred, y_train)

    y_train_inferred, y_test_inferred = pipeline.apply_label_inference(
        y_train, y_test, model=model, X_train=x_train_inferred, X_test=x_test_inferred
    )
    # Coerção de rótulos pós-inferência
    y_train_inferred = _coerce_numeric_labels(y_train_inferred)
    y_test_inferred = _coerce_numeric_labels(y_test_inferred)

    experiment = Experiment(model, dataset)
    metrics = experiment.run(x_train_inferred, x_test_inferred, y_train_inferred, y_test_inferred)

    report_data("Evaluation metrics (with inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.JSON_RESULT, name_output=f"{name_output}-with-inference")


def run() -> None:
    """Runs the complete experiment suite for FastICA(+KMeans)."""
    run_fastica_without_inference()
    run_fastica_with_inference()


if __name__ == "__main__":
    run()
