"""Experiment script for MLPModel on the 20 Newsgroups dataset."""

from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.models.mlp_model import MLPModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


def run():
    """Run MLPModel experiments on the 20 Newsgroups dataset.

    Executes both baseline and inference experiments using standard configurations.
    """
    return run_experiment_by_model(MLPModel, dataset_name=SklearnDatasetName.NEWSGROUPS_20)


if __name__ == "__main__":
    run()
