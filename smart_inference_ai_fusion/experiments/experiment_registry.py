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


# Registry of experiment configurations for the wine dataset
WINE_EXPERIMENTS = {
    AgglomerativeClusteringModel: ExperimentConfig(
        model_class=AgglomerativeClusteringModel,
        model_params={"n_clusters": 3},  # Wine has 3 classes
    ),
    FastICAModel: ExperimentConfig(
        model_class=FastICAModel,
        model_params={"n_components": 3, "random_state": 42},
    ),
    GaussianMixtureModel: ExperimentConfig(
        model_class=GaussianMixtureModel,
        model_params={"n_components": 3, "random_state": 42},
    ),
    GradientBoostingModel: ExperimentConfig(
        model_class=GradientBoostingModel,
        model_params={"n_estimators": 100, "random_state": 42},
    ),
    MiniBatchKMeansModel: ExperimentConfig(
        model_class=MiniBatchKMeansModel,
        model_params={"n_clusters": 3, "random_state": 42, "batch_size": 50},  # Smaller batch for smaller dataset
    ),
    MLPModel: ExperimentConfig(
        model_class=MLPModel,
        model_params={"hidden_layer_sizes": (50,), "random_state": 42, "max_iter": 500},  # Smaller hidden layer
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
            "n_clusters": 3,
            "random_state": 42,
            "affinity": "nearest_neighbors",
            "n_neighbors": 5,  # Smaller for wine dataset
            "assign_labels": "discretize",
            "n_init": 1,
        },
    ),
}


# Registry of experiment configurations for the California Housing dataset
CALIFORNIA_HOUSING_EXPERIMENTS = {
    GradientBoostingModel: ExperimentConfig(
        model_class=GradientBoostingModel,
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
    # Clustering algorithms for market segmentation
    AgglomerativeClusteringModel: ExperimentConfig(
        model_class=AgglomerativeClusteringModel,
        model_params={"n_clusters": 5},  # Market segments
    ),
    MiniBatchKMeansModel: ExperimentConfig(
        model_class=MiniBatchKMeansModel,
        model_params={"n_clusters": 5, "random_state": 42, "batch_size": 200},
    ),
    GaussianMixtureModel: ExperimentConfig(
        model_class=GaussianMixtureModel,
        model_params={"n_components": 5, "random_state": 42},
    ),
}


# Registry of experiment configurations for the LFW People dataset  
LFW_PEOPLE_EXPERIMENTS = {
    RandomForestClassifierModel: ExperimentConfig(
        model_class=RandomForestClassifierModel,
        model_params={"n_estimators": 50, "random_state": 42},  # Reduced for high-dim data
    ),
    MLPModel: ExperimentConfig(
        model_class=MLPModel,
        model_params={"hidden_layer_sizes": (100, 50), "random_state": 42, "max_iter": 200},
    ),
    RidgeModel: ExperimentConfig(
        model_class=RidgeModel,
        model_params={"alpha": 10.0, "random_state": 42},  # Higher regularization
    ),
    # Dimensionality reduction
    FastICAModel: ExperimentConfig(
        model_class=FastICAModel,
        model_params={"n_components": 50, "random_state": 42},  # Reduce from high dimensions
    ),
    # Additional models for 10 experiments total
    GradientBoostingModel: ExperimentConfig(
        model_class=GradientBoostingModel,
        model_params={
            "n_estimators": 10,  # Extremamente reduzido para alta dimensionalidade
            "random_state": 42,
            "max_depth": 5,  # Muito raso para evitar overfitting
            "learning_rate": 0.6,  # Learning rate maior para convergir mais rápido
            "subsample": 0.9,  # Subsampling agressivo
            "max_features": "sqrt",  # Reduzir features por árvore
        },
    ),
    RandomForestRegressorModel: ExperimentConfig(
        model_class=RandomForestRegressorModel,
        model_params={"n_estimators": 50, "random_state": 42},  # Reduced for high-dim data
    ),
    # Clustering for face analysis
    AgglomerativeClusteringModel: ExperimentConfig(
        model_class=AgglomerativeClusteringModel,
        model_params={"n_clusters": 10},  # Multiple people clusters
    ),
    MiniBatchKMeansModel: ExperimentConfig(
        model_class=MiniBatchKMeansModel,
        model_params={"n_clusters": 10, "random_state": 42, "batch_size": 100},  # Face clustering
    ),
    GaussianMixtureModel: ExperimentConfig(
        model_class=GaussianMixtureModel,
        model_params={"n_components": 10, "random_state": 42},  # Face mixture modeling
    ),
    SpectralClusteringModel: ExperimentConfig(
        model_class=SpectralClusteringModel,
        model_params={
            "n_clusters": 10,
            "random_state": 42,
            "affinity": "nearest_neighbors",
            "n_neighbors": 10,
            "assign_labels": "discretize",
            "n_init": 1,
        },
    ),
}


# Registry of experiment configurations for the Make Moons dataset
MAKE_MOONS_EXPERIMENTS = {
    SpectralClusteringModel: ExperimentConfig(
        model_class=SpectralClusteringModel,
        model_params={
            "n_clusters": 2,
            "random_state": 42,
            "affinity": "nearest_neighbors", 
            "n_neighbors": 10,
            "assign_labels": "discretize",
            "n_init": 1,
        },
    ),
    AgglomerativeClusteringModel: ExperimentConfig(
        model_class=AgglomerativeClusteringModel,
        model_params={"n_clusters": 2},
    ),
    MiniBatchKMeansModel: ExperimentConfig(
        model_class=MiniBatchKMeansModel,
        model_params={"n_clusters": 2, "random_state": 42, "batch_size": 100},
    ),
    GaussianMixtureModel: ExperimentConfig(
        model_class=GaussianMixtureModel,
        model_params={"n_components": 2, "random_state": 42},
    ),
    # Classification algorithms to compare with clustering
    RandomForestClassifierModel: ExperimentConfig(
        model_class=RandomForestClassifierModel,
        model_params={"n_estimators": 100, "random_state": 42},
    ),
    MLPModel: ExperimentConfig(
        model_class=MLPModel,
        model_params={"hidden_layer_sizes": (50,), "random_state": 42, "max_iter": 500},
    ),
    # Additional models for 10 experiments total
    GradientBoostingModel: ExperimentConfig(
        model_class=GradientBoostingModel,
        model_params={"n_estimators": 50, "random_state": 42},  # Reduced for 2D dataset
    ),
    FastICAModel: ExperimentConfig(
        model_class=FastICAModel,
        model_params={"n_components": 2, "random_state": 42},  # 2 components for 2D dataset
    ),
    RidgeModel: ExperimentConfig(
        model_class=RidgeModel,
        model_params={"alpha": 1.0, "random_state": 42},
    ),
    RandomForestRegressorModel: ExperimentConfig(
        model_class=RandomForestRegressorModel,
        model_params={"n_estimators": 50, "random_state": 42},  # Reduced for 2D dataset
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
    # Get the appropriate registry based on dataset
    if dataset_name == SklearnDatasetName.WINE:
        registry = WINE_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.CALIFORNIA_HOUSING:
        registry = CALIFORNIA_HOUSING_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.LFW_PEOPLE:
        registry = LFW_PEOPLE_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.MAKE_MOONS:
        registry = MAKE_MOONS_EXPERIMENTS
    else:
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
    # Get the appropriate registry based on dataset
    if dataset_name == SklearnDatasetName.WINE:
        registry = WINE_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.CALIFORNIA_HOUSING:
        registry = CALIFORNIA_HOUSING_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.LFW_PEOPLE:
        registry = LFW_PEOPLE_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.MAKE_MOONS:
        registry = MAKE_MOONS_EXPERIMENTS
    else:
        registry = DIGITS_EXPERIMENTS
    
    # Get configuration directly using model class as key
    if model_class not in registry:
        available_models = [cls.__name__ for cls in registry]
        raise ValueError(
            f"Model '{model_class.__name__}' not found in registry. "
            f"Available: {', '.join(available_models)}"
        )

    config = registry[model_class]
    # Use custom params if provided, otherwise use registry defaults
    params = model_params if model_params is not None else config.model_params

    return run_standard_experiment(
        model_class=model_class,
        model_name=model_class.__name__,
        dataset_source=dataset_source,
        dataset_name=dataset_name,
        model_params=params,
    )
