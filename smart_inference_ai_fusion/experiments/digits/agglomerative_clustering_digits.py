"""Experiment script for AgglomerativeClusteringModel on the Digits dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.agglomerative_clustering_model import (
    AgglomerativeClusteringModel,
)


def run():
    """Run AgglomerativeClusteringModel experiments on the Digits dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(AgglomerativeClusteringModel)


if __name__ == "__main__":
    run()
