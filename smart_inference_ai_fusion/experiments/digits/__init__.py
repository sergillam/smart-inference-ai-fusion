"""Digits experiments package.

This package organizes and runs experiments for the Digits dataset.
"""

from smart_inference_ai_fusion.utils.report import report_data
from smart_inference_ai_fusion.utils.types import ReportMode

from .agglomerative_clustering_digits import run as run_agglomerative_clustering_digits
from .fastica_digits import run as run_fastica_digits
from .gaussian_mixture_digits import run as run_gaussian_mixture_digits
from .gradient_boosting_digits import run as run_gradient_boosting
from .minibatch_kmeans_digits import run as run_minibatch_kmeans
from .mlp_digits import run as run_mlp
from .random_forest_classifier_digits import run as run_random_forest
from .random_forest_regressor_digits import run as run_rf_regressor
from .ridge_digits import run as run_ridge
from .spectral_clustering_digits import run as run_spectral_clustering


def run_all():
    """Runs all the experiments defined for the Digits dataset.

    This function sequentially calls the execution scripts for each of the
    machine learning models applied to the Digits database.
    """
    report_data("=== Executando Todos os Experimentos com a Base DIGITS ===", mode=ReportMode.PRINT)
    run_random_forest()
    run_gradient_boosting()
    run_mlp()
    run_ridge()
    run_rf_regressor()
    run_minibatch_kmeans()
    run_spectral_clustering()
    run_gaussian_mixture_digits()
    run_agglomerative_clustering_digits()
    run_fastica_digits()
    report_data("=== Todos os Experimentos Concluídos ===", mode=ReportMode.PRINT)
