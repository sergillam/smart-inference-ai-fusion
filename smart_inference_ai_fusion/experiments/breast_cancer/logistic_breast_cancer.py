"""Experiment script for LogisticRegressionModel on the Breast Cancer dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.logistic_regression_model import LogisticRegressionModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run LogisticRegressionModel experiments on the Breast Cancer dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(LogisticRegressionModel, dataset_name=SklearnDatasetName.BREAST_CANCER)


if __name__ == "__main__":
    run()