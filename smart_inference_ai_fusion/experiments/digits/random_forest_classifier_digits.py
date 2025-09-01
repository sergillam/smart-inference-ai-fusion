"""Experiment script for RandomForestClassifierModel on the Digits dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.random_forest_classifier_model import (
    RandomForestClassifierModel,
)


def run():
    """Run RandomForestClassifierModel experiments on the Digits dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(RandomForestClassifierModel)


if __name__ == "__main__":
    run()
