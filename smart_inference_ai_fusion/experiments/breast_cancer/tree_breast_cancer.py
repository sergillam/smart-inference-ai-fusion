"""Experiment script for DecisionTreeModel on the Breast Cancer dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.tree_model import DecisionTreeModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run DecisionTreeModel experiments on the Breast Cancer dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(DecisionTreeModel, dataset_name=SklearnDatasetName.BREAST_CANCER)


if __name__ == "__main__":
    run()