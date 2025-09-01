"""Experiment script for FastICAModel on the Digits dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.models.fastica_model import FastICAModel


def run():
    """Run FastICAModel experiments on the Digits dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(FastICAModel)


if __name__ == "__main__":
    run()
