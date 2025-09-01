"""Experiment script for MLPModel on the Digits dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.models.mlp_model import MLPModel


def run():
    """Run MLPModel experiments on the Digits dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(MLPModel)


if __name__ == "__main__":
    run()
