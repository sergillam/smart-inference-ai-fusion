"""Experiment script for GradientBoostingModel on the LFW People dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.gradient_boosting_model import GradientBoostingModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run GradientBoostingModel experiments on the LFW People dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(
        GradientBoostingModel, dataset_name=SklearnDatasetName.LFW_PEOPLE
    )


if __name__ == "__main__":
    run()
