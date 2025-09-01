"""Experiment script for AgglomerativeClusteringModel on the Wine dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.agglomerative_clustering_model import AgglomerativeClusteringModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run AgglomerativeClusteringModel experiments on the Wine dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(
        AgglomerativeClusteringModel, 
        dataset_name=SklearnDatasetName.WINE
    )


if __name__ == "__main__":
    run()
