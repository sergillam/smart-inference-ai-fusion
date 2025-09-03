"""Experiment script for SpectralClusteringModel on the Make Moons dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.spectral_clustering_model import SpectralClusteringModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run SpectralClusteringModel experiments on the Make Moons dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(
        SpectralClusteringModel, dataset_name=SklearnDatasetName.MAKE_MOONS
    )


if __name__ == "__main__":
    run()
