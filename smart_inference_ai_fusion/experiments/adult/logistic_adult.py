"""
Adult Dataset Logistic Regression Experiment
============================================

This experiment applies logistic regression to the Adult dataset
with formal verification and controlled perturbations.
"""

from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.models.logistic_regression_model import LogisticRegressionModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run Adult dataset logistic regression experiment."""
    return run_experiment_by_model(
        LogisticRegressionModel,
        dataset_name=SklearnDatasetName.ADULT
    )


if __name__ == "__main__":
    run()