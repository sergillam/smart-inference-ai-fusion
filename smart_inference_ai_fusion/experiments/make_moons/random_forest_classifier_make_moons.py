"""Experiment script for RandomForestClassifierModel on the Make Moons dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.random_forest_classifier_model import (
    RandomForestClassifierModel,
)
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run RandomForestClassifierModel experiments on the Make Moons dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(
        RandomForestClassifierModel, dataset_name=SklearnDatasetName.MAKE_MOONS
    )


if __name__ == "__main__":
    run()
