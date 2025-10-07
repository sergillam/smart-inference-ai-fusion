"""MiniBatch KMeans experiment for Make Blobs dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.minibatch_kmeans_model import MiniBatchKMeansModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run Make Blobs dataset MiniBatchKMeans experiment."""
    return run_experiment_by_model(MiniBatchKMeansModel, dataset_name=SklearnDatasetName.MAKE_BLOBS)


if __name__ == "__main__":
    run()
