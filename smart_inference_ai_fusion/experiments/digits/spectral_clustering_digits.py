"""Experiment script for SpectralClustering on the Digits dataset."""

from __future__ import annotations

import numpy as np

from smart_inference_ai_fusion.core.experiment import Experiment
from smart_inference_ai_fusion.datasets.factory import DatasetFactory
from smart_inference_ai_fusion.inference.engine.param_runner import (
    ParameterInferenceEngine,
)
from smart_inference_ai_fusion.inference.pipeline.inference_pipeline import (
    InferencePipeline,
)
from smart_inference_ai_fusion.models.spectral_clustering_model import (
    SpectralClusteringModel,
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

# Valores válidos para SpectralClustering (sklearn 1.7.1)
_VALID_AFFINITIES = {
    "laplacian",
    "precomputed",
    "precomputed_nearest_neighbors",
    "polynomial",
    "poly",
    "cosine",
    "rbf",
    "nearest_neighbors",
    "chi2",
    "additive_chi2",
    "linear",
    "sigmoid",
}
_VALID_ASSIGN = {"kmeans", "discretize", "cluster_qr"}


def _coerce_numeric_labels(y):
    """Converte rótulos para inteiros quando possível, mantendo caso contrário."""
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        return y
    try:
        y_int = y.astype(int)
        if np.all(y_int.astype(str) == y):
            return y_int
    except (ValueError, TypeError):
        pass
    return y


def _sanitize_spectral_params(params: dict) -> dict:
    """Limpa parâmetros potencialmente inválidos vindos da inferência."""
    p = dict(params)

    # enums robustos
    if p.get("affinity") not in _VALID_AFFINITIES:
        p["affinity"] = "nearest_neighbors"
    if p.get("assign_labels") not in _VALID_ASSIGN:
        p["assign_labels"] = "kmeans"

    # mínimos razoáveis
    if "n_neighbors" in p:
        try:
            p["n_neighbors"] = int(max(2, int(p["n_neighbors"])))
        except (ValueError, TypeError):
            p["n_neighbors"] = 10
    else:
        p["n_neighbors"] = 10

    if "random_state" in p and p["random_state"] is not None:
        try:
            p["random_state"] = int(p["random_state"])
        except (ValueError, TypeError):
            p["random_state"] = 42
    else:
        p["random_state"] = 42

    # n_neighbors só faz sentido com affinities baseadas em vizinhos
    if p.get("affinity") not in {"nearest_neighbors", "precomputed_nearest_neighbors"}:
        p.pop("n_neighbors", None)

    return p


def _lock_structure_to_dataset(p: dict, y_train) -> dict:
    """Garante que n_clusters == nº de classes observadas (quando faz sentido)."""
    p = dict(p)
    try:
        n_classes = int(len(np.unique(y_train)))
        p["n_clusters"] = max(2, n_classes)
    except (ValueError, TypeError):
        p["n_clusters"] = max(2, int(p.get("n_clusters", 10)))
    return p


# pylint: disable=duplicate-code
def run_spectral_clustering_without_inference() -> None:
    """Run the SpectralClustering baseline (no inference or perturbations)."""
    model_class = SpectralClusteringModel
    dataset_name = SklearnDatasetName.DIGITS
    name_output = generate_experiment_filename(model_class, dataset_name)

    report_data("=== SpectralClustering WITHOUT INFERENCE ===", mode=ReportMode.PRINT)

    base_params = {
        "n_clusters": 10,
        "random_state": 42,
        "affinity": "nearest_neighbors",
        "n_neighbors": 10,
        "assign_labels": "kmeans",
    }

    filtered_params = filter_sklearn_params(base_params, SpectralClusteringModel)
    model = model_class(**filtered_params)

    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=dataset_name)
    x_train, x_test, y_train, y_test = dataset.load_data()

    experiment = Experiment(model, dataset)
    metrics = experiment.run(x_train, x_test, y_train, y_test)

    report_data("Evaluation metrics (no inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.JSON_RESULT, name_output=f"{name_output}-no-inference")


def run_spectral_clustering_with_inference() -> None:
    """Run the SpectralClustering model with data/param/label inference."""
    model_class = SpectralClusteringModel
    dataset_name = SklearnDatasetName.DIGITS
    name_output = generate_experiment_filename(model_class, dataset_name)

    report_data(
        "=== SpectralClustering WITH INFERENCE (data + param + label) ===",
        mode=ReportMode.PRINT,
    )

    base_params = {
        "n_clusters": 10,
        "random_state": 42,
        "affinity": "nearest_neighbors",
        "n_neighbors": 10,
        "assign_labels": "kmeans",
    }

    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=dataset_name)

    # --- ruídos ---
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

    # --- parâmetros perturbados + saneamento + travamento à estrutura ---
    param_engine = ParameterInferenceEngine(config=param_noise_config)
    perturbed = param_engine.apply(base_params)
    perturbed = _sanitize_spectral_params(perturbed)
    perturbed = _lock_structure_to_dataset(perturbed, y_train)

    report_data(f"Perturbed parameters: {perturbed}", mode=ReportMode.PRINT)
    report_data(
        param_engine.export_log(),
        mode=ReportMode.JSON_LOG,
        name_output=f"{name_output}-param-perturb",
    )

    filtered_params = filter_sklearn_params(perturbed, SpectralClusteringModel)
    model = SpectralClusteringModel(**filtered_params)

    # --- inferência de dados ---
    x_train_inf, x_test_inf = pipeline.apply_data_inference(x_train, x_test)
    model.fit(x_train_inf, y_train)

    # --- inferência de labels ---
    y_train_inf, y_test_inf = pipeline.apply_label_inference(
        y_train, y_test, model=model, X_train=x_train_inf, X_test=x_test_inf
    )
    y_train_inf = _coerce_numeric_labels(y_train_inf)

    # 🔒 para avaliação justa, NÃO ruidar y_test: usamos rótulos originais
    # (caso queira avaliar com y_test ruidoso também, salve as duas versões)
    y_test_inf = _coerce_numeric_labels(y_test)

    # --- avaliação ---
    experiment = Experiment(model, dataset)
    metrics = experiment.run(x_train_inf, x_test_inf, y_train_inf, y_test_inf)

    report_data("Evaluation metrics (with inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.JSON_RESULT, name_output=f"{name_output}-with-inference")


def run() -> None:
    """Runs the complete experiment suite for the SpectralClustering."""
    run_spectral_clustering_without_inference()
    run_spectral_clustering_with_inference()


if __name__ == "__main__":
    run()
