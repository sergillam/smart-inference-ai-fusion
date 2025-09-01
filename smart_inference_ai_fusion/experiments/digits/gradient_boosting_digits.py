"""Experiment script for GradientBoostingModel on the Digits dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.models.gradient_boosting_model import GradientBoostingModel


def run():
    """Run GradientBoostingModel experiments on the Digits dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(GradientBoostingModel)


if __name__ == "__main__":
    run()
