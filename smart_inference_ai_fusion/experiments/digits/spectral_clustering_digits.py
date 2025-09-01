"""Experiment script for SpectralClusteringModel on the Digits dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.models.spectral_clustering_model import SpectralClusteringModel


def run():
    """Run SpectralClusteringModel experiments on the Digits dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(SpectralClusteringModel)


if __name__ == "__main__":
    run()
