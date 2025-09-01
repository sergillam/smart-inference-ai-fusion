"""Experiment registry for standardized experiment configurations.

This module provides a centralized registry of experiment configurations
to eliminate code duplication across experiment scripts.
"""

from typing import Any, Dict, Type

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.experiments.common import run_standard_experiment
from smart_inference_ai_fusion.models.agglomerative_clustering_model import (
    AgglomerativeClusteringModel,
)
from smart_inference_ai_fusion.models.fastica_model import FastICAModel
from smart_inference_ai_fusion.models.gaussian_mixture_model import GaussianMixtureModel
from smart_inference_ai_fusion.models.gradient_boosting_model import GradientBoostingModel
from smart_inference_ai_fusion.models.minibatch_kmeans_model import MiniBatchKMeansModel
from smart_inference_ai_fusion.models.mlp_model import MLPModel
from smart_inference_ai_fusion.models.random_forest_classifier_model import (
    RandomForestClassifierModel,
)
from smart_inference_ai_fusion.models.random_forest_regressor_model import (
    RandomForestRegressorModel,
)
from smart_inference_ai_fusion.models.ridge_model import RidgeModel
from smart_inference_ai_fusion.models.spectral_clustering_model import SpectralClusteringModel
from smart_inference_ai_fusion.utils.types import DatasetSourceType, SklearnDatasetName


class ExperimentConfig:
    """Configuration for a standardized experiment."""

    def __init__(
        self,
        model_class: Type[BaseModel],
        model_params: Dict[str, Any],
        dataset_source: DatasetSourceType = DatasetSourceType.SKLEARN,
        dataset_name: SklearnDatasetName = SklearnDatasetName.DIGITS,
    ):
        """Initialize experiment configuration.

        Args:
            model_class: The model class to instantiate.
            model_params: Parameters to pass to the model constructor.
            dataset_source: Source type for the dataset.
            dataset_name: Name of the dataset.
        """
        self.model_class = model_class
        self.model_name = model_class.__name__  # Auto-generate from class name
        self.model_params = model_params
        self.dataset_source = dataset_source
        self.dataset_name = dataset_name


# Registry of experiment configurations for the digits dataset
DIGITS_EXPERIMENTS = {
    AgglomerativeClusteringModel: ExperimentConfig(
        model_class=AgglomerativeClusteringModel,
        model_params={"n_clusters": 10},
    ),
    FastICAModel: ExperimentConfig(
        model_class=FastICAModel,
        model_params={"n_components": 10, "random_state": 42},
    ),
    GaussianMixtureModel: ExperimentConfig(
        model_class=GaussianMixtureModel,
        model_params={"n_components": 10, "random_state": 42},
    ),
    GradientBoostingModel: ExperimentConfig(
        model_class=GradientBoostingModel,
        model_params={"n_estimators": 100, "random_state": 42},
    ),
    MiniBatchKMeansModel: ExperimentConfig(
        model_class=MiniBatchKMeansModel,
        model_params={"n_clusters": 10, "random_state": 42, "batch_size": 100},
    ),
    MLPModel: ExperimentConfig(
        model_class=MLPModel,
        model_params={"hidden_layer_sizes": (100,), "random_state": 42, "max_iter": 500},
    ),
    RandomForestClassifierModel: ExperimentConfig(
        model_class=RandomForestClassifierModel,
        model_params={"n_estimators": 100, "random_state": 42},
    ),
    RandomForestRegressorModel: ExperimentConfig(
        model_class=RandomForestRegressorModel,
        model_params={"n_estimators": 100, "random_state": 42},
    ),
    RidgeModel: ExperimentConfig(
        model_class=RidgeModel,
        model_params={"alpha": 1.0, "random_state": 42},
    ),
    SpectralClusteringModel: ExperimentConfig(
        model_class=SpectralClusteringModel,
        model_params={
            "n_clusters": 10,
            "random_state": 42,
            "affinity": "nearest_neighbors",
            "n_neighbors": 10,
            "assign_labels": "discretize",
            "n_init": 1,  # Reduce iterations for faster execution
        },
    ),
}


def run_experiment_by_name(
    experiment_name: str,
    dataset_name: SklearnDatasetName = SklearnDatasetName.DIGITS,
    dataset_source: DatasetSourceType = DatasetSourceType.SKLEARN,
):
    """Run a registered experiment by name.

    Args:
        experiment_name: Name of the experiment in the registry.
        dataset_name: Dataset to run the experiment on.
        dataset_source: Source type for the dataset (sklearn or external).

    Returns:
        Result of the experiment.

    Raises:
        KeyError: If experiment_name is not found in the registry.
    """
    # Get the base config and update dataset if needed
    if dataset_name == SklearnDatasetName.DIGITS:
        registry = DIGITS_EXPERIMENTS
    else:
        # For other datasets, use digits config but update dataset_name
        registry = DIGITS_EXPERIMENTS

    if experiment_name not in registry:
        available = ", ".join(registry.keys())
        raise KeyError(f"Experiment '{experiment_name}' not found. Available: {available}")

    config = registry[experiment_name]

    return run_standard_experiment(
        model_class=config.model_class,
        model_name=config.model_name,
        dataset_source=dataset_source,
        dataset_name=dataset_name,
        model_params=config.model_params,
    )


def run_experiment_by_model(
    model_class: Type[BaseModel],
    dataset_name: SklearnDatasetName = SklearnDatasetName.DIGITS,
    dataset_source: DatasetSourceType = DatasetSourceType.SKLEARN,
    model_params: Dict[str, Any] = None,
):
    """Run an experiment by passing the model class directly.

    Args:
        model_class: The model class to run (e.g., AgglomerativeClusteringModel).
        dataset_name: Dataset to run the experiment on.
        dataset_source: Source type for the dataset (sklearn or external).
        model_params: Custom model parameters (optional).

    Returns:
        Result of the experiment.

    Raises:
        ValueError: If model_class is not found in any registry.
    """
    # Get configuration directly using model class as key
    if model_class not in DIGITS_EXPERIMENTS:
        available_models = [cls.__name__ for cls in DIGITS_EXPERIMENTS]
        raise ValueError(
            f"Model '{model_class.__name__}' not found in registry. "
            f"Available: {', '.join(available_models)}"
        )

    config = DIGITS_EXPERIMENTS[model_class]
    # Use custom params if provided, otherwise use registry defaults
    params = model_params if model_params is not None else config.model_params

    return run_standard_experiment(
        model_class=model_class,
        model_name=model_class.__name__,
        dataset_source=dataset_source,
        dataset_name=dataset_name,
        model_params=params,
    )
