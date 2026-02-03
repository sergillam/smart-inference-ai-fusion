"""Logistic Regression experiment for Make Blobs dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.logistic_regression_model import LogisticRegressionModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run Make Blobs dataset logistic regression experiment."""
    return run_experiment_by_model(
        LogisticRegressionModel, dataset_name=SklearnDatasetName.MAKE_BLOBS
    )


if __name__ == "__main__":
    run()
