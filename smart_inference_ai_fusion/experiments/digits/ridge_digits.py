"""Experiment script for RidgeModel on the Digits dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.models.ridge_model import RidgeModel


def run():
    """Run RidgeModel experiments on the Digits dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(RidgeModel)


if __name__ == "__main__":
    run()
