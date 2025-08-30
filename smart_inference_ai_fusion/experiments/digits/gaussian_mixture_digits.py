"""Experiment script for Gaussian Mixture on the Digits dataset."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from smart_inference_ai_fusion.core.experiment import Experiment
from smart_inference_ai_fusion.datasets.factory import DatasetFactory
from smart_inference_ai_fusion.inference.engine.param_runner import ParameterInferenceEngine
from smart_inference_ai_fusion.inference.pipeline.inference_pipeline import InferencePipeline
from smart_inference_ai_fusion.models.gaussian_mixture_model import GaussianMixtureModel
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

# ------------------------------
# Helpers para sanitização
# ------------------------------

_VALID_COV = {"full", "tied", "diag", "spherical"}
_VALID_INIT = {"kmeans", "k-means++", "random"}


def _safe_int(val, default):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _sanitize_enums(p: dict) -> None:
    cov = p.get("covariance_type")
    if cov not in _VALID_COV:
        p["covariance_type"] = "full"

    initp = p.get("init_params")
    if initp not in _VALID_INIT:
        # scikit usa 'kmeans' e 'k-means++' dependendo da versão;
        # manter 'kmeans' que é aceito em versões atuais.
        p["init_params"] = "kmeans"


def _sanitize_ints(p: dict) -> None:
    # n_components será travado mais adiante pelo dataset
    if "n_init" in p:
        p["n_init"] = _clamp(_safe_int(p["n_init"], 1), 1, 50)
    else:
        p["n_init"] = 1

    if "max_iter" in p:
        p["max_iter"] = _clamp(_safe_int(p["max_iter"], 200), 10, 1000)
    else:
        p["max_iter"] = 200

    if "random_state" in p and p["random_state"] is not None:
        p["random_state"] = _safe_int(p["random_state"], 42)
    else:
        p["random_state"] = 42


def _sanitize_floats(p: dict) -> None:
    if "reg_covar" in p:
        p["reg_covar"] = _clamp(_safe_float(p["reg_covar"], 1e-6), 0.0, 1e-1)
    else:
        p["reg_covar"] = 1e-6


def _lock_components_to_labels(p: dict, y_train) -> None:
    """Garante n_components = nº de classes observadas (mínimo 2)."""
    try:
        n_classes = int(len(np.unique(y_train)))
    except (ValueError, TypeError):
        n_classes = _safe_int(p.get("n_components", 10), 10)
    p["n_components"] = max(2, n_classes)


def _sanitize_gmm_params(params: dict, y_train) -> dict:
    """Sanitiza e torna robustos os parâmetros para GaussianMixture."""
    p = dict(params) if params is not None else {}
    _sanitize_enums(p)
    _sanitize_ints(p)
    _sanitize_floats(p)
    _lock_components_to_labels(p, y_train)
    return p


def _stabilize_for_noisy_data(p: dict) -> dict:
    """Ajustes extras para rodar GMM em dados ruidosos."""
    q = dict(p)
    q["reg_covar"] = max(float(q.get("reg_covar", 1e-6)), 1e-4)  # mais PD
    q["n_init"] = max(int(q.get("n_init", 1)), 5)  # mais inicializações
    return q


# ------------------------------
# Experimentos
# ------------------------------


def run_gaussian_mixture_without_inference() -> None:
    """Baseline (sem inferência/perturbações)."""
    model_class = GaussianMixtureModel
    dataset_name = SklearnDatasetName.DIGITS
    name_output = generate_experiment_filename(model_class, dataset_name)

    report_data("=== GaussianMixture WITHOUT INFERENCE ===", mode=ReportMode.PRINT)

    base_params = {
        "n_components": 10,
        "covariance_type": "full",
        "init_params": "kmeans",
        "random_state": 42,
        "n_init": 1,
        "max_iter": 200,
        "reg_covar": 1e-6,
    }

    filtered_params = filter_sklearn_params(base_params, GaussianMixtureModel)
    model = model_class(**filtered_params)

    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=dataset_name)
    X_train, X_test, y_train, y_test = dataset.load_data()

    experiment = Experiment(model, dataset)
    metrics = experiment.run(X_train, X_test, y_train, y_test)

    report_data("Evaluation metrics (no inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.JSON_RESULT, name_output=f"{name_output}-no-inference")


def run_gaussian_mixture_with_inference() -> None:
    """Executa GMM com inferência (ruídos em dados, labels e params)."""
    model_class = GaussianMixtureModel
    dataset_name = SklearnDatasetName.DIGITS
    name_output = generate_experiment_filename(model_class, dataset_name)

    report_data(
        "=== GaussianMixture WITH INFERENCE (data + param + label) ===", mode=ReportMode.PRINT
    )
    base_params = {
        "n_components": 10,
        "covariance_type": "full",
        "init_params": "kmeans",
        "random_state": 42,
        "n_init": 1,
        "max_iter": 200,
        "reg_covar": 1e-6,
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

    X_train, X_test, y_train, y_test = dataset.load_data()

    pipeline = InferencePipeline(
        data_noise_config=data_noise_config,
        label_noise_config=label_noise_config,
        X_train=X_train,
    )

    # Perturba params e sanitiza
    param_engine = ParameterInferenceEngine(config=param_noise_config)
    perturbed = param_engine.apply(base_params)
    perturbed = _sanitize_gmm_params(perturbed, y_train)
    perturbed = _stabilize_for_noisy_data(perturbed)

    report_data(f"Perturbed parameters: {perturbed}", mode=ReportMode.PRINT)
    report_data(
        param_engine.export_log(),
        mode=ReportMode.JSON_LOG,
        name_output=f"{name_output}-param-perturb",
    )

    filtered_params = filter_sklearn_params(perturbed, GaussianMixtureModel)
    model = GaussianMixtureModel(**filtered_params)

    # Ruído nos dados + estabilização numérica: float64 + StandardScaler
    x_train_inf, x_test_inf = pipeline.apply_data_inference(X_train, X_test)
    x_train_inf = np.asarray(x_train_inf, dtype=np.float64)
    x_test_inf = np.asarray(x_test_inf, dtype=np.float64)

    scaler = StandardScaler()
    x_train_inf = scaler.fit_transform(x_train_inf)
    x_test_inf = scaler.transform(x_test_inf)

    model.fit(x_train_inf, y_train)

    # Ruído nos rótulos
    y_train_inf, y_test_inf = pipeline.apply_label_inference(
        y_train, y_test, model=model, X_train=x_train_inf, X_test=x_test_inf
    )

    experiment = Experiment(model, dataset)
    metrics = experiment.run(x_train_inf, x_test_inf, y_train_inf, y_test_inf)

    report_data("Evaluation metrics (with inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.JSON_RESULT, name_output=f"{name_output}-with-inference")


def run() -> None:
    """Roda os dois cenários (sem/como inferência)."""
    run_gaussian_mixture_without_inference()
    run_gaussian_mixture_with_inference()


if __name__ == "__main__":
    run()
