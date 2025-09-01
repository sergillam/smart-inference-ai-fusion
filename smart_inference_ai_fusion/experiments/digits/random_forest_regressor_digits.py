"""Experiment script for RandomForestRegressorModel on the Digits dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.random_forest_regressor_model import (
    RandomForestRegressorModel,
)


def run():
    """Run RandomForestRegressorModel experiments on the Digits dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(RandomForestRegressorModel)


if __name__ == "__main__":
    run()
