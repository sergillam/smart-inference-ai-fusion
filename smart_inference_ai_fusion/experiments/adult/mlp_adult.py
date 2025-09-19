"""
Adult Dataset MLP Experiment
============================

This experiment applies Multi-Layer Perceptron to the Adult dataset
with formal verification and controlled perturbations.
"""

from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.models.mlp_model import MLPModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run Adult dataset MLP experiment."""
    return run_experiment_by_model(
        MLPModel,
        dataset_name=SklearnDatasetName.ADULT
    )


if __name__ == "__main__":
    run()