"""Experiment script for MiniBatchKMeansModel on the LFW People dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.minibatch_kmeans_model import MiniBatchKMeansModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run MiniBatchKMeansModel experiments on the LFW People dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(MiniBatchKMeansModel, dataset_name=SklearnDatasetName.LFW_PEOPLE)


if __name__ == "__main__":
    run()
