"""Experiment script for MiniBatchKMeansModel on the 20 Newsgroups dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.models.minibatch_kmeans_model import MiniBatchKMeansModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run MiniBatchKMeansModel experiments on the 20 Newsgroups dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(
        MiniBatchKMeansModel, dataset_name=SklearnDatasetName.NEWSGROUPS_20
    )


if __name__ == "__main__":
    run()
