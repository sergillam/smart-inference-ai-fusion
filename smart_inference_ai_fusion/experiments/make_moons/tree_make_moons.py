"""Decision Tree experiment for Make Moons dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import (
    run_experiment_by_model,
)
from smart_inference_ai_fusion.models.tree_model import DecisionTreeModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run Make Moons dataset decision tree experiment."""
    return run_experiment_by_model(DecisionTreeModel, dataset_name=SklearnDatasetName.MAKE_MOONS)


if __name__ == "__main__":
    run()
