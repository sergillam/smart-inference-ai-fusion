"""Digits experiments package.

This package organizes and runs experiments for the Digits dataset.
"""

from src.utils.report import report_data
from utils.types import ReportMode
from .random_forest_classifier_digits import run as run_random_forest
from .gradient_boosting_digits import run as run_gradient_boosting
from .mlp_digits import run as run_mlp
from .ridge_digits import run as run_ridge
from .random_forest_regressor_digits import run as run_rf_regressor

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
    report_data("=== Todos os Experimentos Concluídos ===", mode=ReportMode.PRINT)
