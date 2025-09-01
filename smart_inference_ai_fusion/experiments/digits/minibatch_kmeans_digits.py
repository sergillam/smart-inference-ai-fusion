"""Experiment script for MiniBatchKMeansModel on the Digits dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.models.minibatch_kmeans_model import MiniBatchKMeansModel


def run():
    """Run MiniBatchKMeansModel experiments on the Digits dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(MiniBatchKMeansModel)


if __name__ == "__main__":
    run()
