"""
Adult Dataset Decision Tree Experiment
======================================

This experiment applies decision tree classification to the Adult dataset
with formal verification and controlled perturbations.
"""

from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.models.tree_model import DecisionTreeModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run Adult dataset decision tree experiment."""
    return run_experiment_by_model(
        DecisionTreeModel,
        dataset_name=SklearnDatasetName.ADULT
    )


if __name__ == "__main__":
    run()