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
        model_params={"n_clusters": 10, "linkage": "ward"},  # 10 digits (0-9) - no random_state
    ),
    FastICAModel: ExperimentConfig(
        model_class=FastICAModel,
        model_params={
            "n_components": 32,
            "random_state": 42,
            "max_iter": 1000,
            "tol": 1e-4,
        },  # Scientific parameters
    ),
    GaussianMixtureModel: ExperimentConfig(
        model_class=GaussianMixtureModel,
        model_params={
            "n_components": 10,
            "random_state": 42,
            "covariance_type": "full",
            "max_iter": 100,
        },  # 10 digits
    ),
    GradientBoostingModel: ExperimentConfig(
        model_class=GradientBoostingModel,
        model_params={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
        },  # Standard parameters
    ),
    MiniBatchKMeansModel: ExperimentConfig(
        model_class=MiniBatchKMeansModel,
        model_params={
            "n_clusters": 10,
            "random_state": 42,
            "batch_size": 256,
            "max_iter": 100,
        },  # 10 digits, reasonable batch
    ),
    MLPModel: ExperimentConfig(
        model_class=MLPModel,
        model_params={
            "hidden_layer_sizes": (100, 50),
            "random_state": 42,
            "max_iter": 500,
            "alpha": 0.001,
        },  # Multi-layer with regularization
    ),
    RandomForestClassifierModel: ExperimentConfig(
        model_class=RandomForestClassifierModel,
        model_params={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "min_samples_split": 5,
        },  # Standard RF parameters
    ),
    RandomForestRegressorModel: ExperimentConfig(
        model_class=RandomForestRegressorModel,
        model_params={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "min_samples_split": 5,
        },
    ),
    RidgeModel: ExperimentConfig(
        model_class=RidgeModel,
        model_params={
            "alpha": 1.0,
            "random_state": 42,
            "max_iter": 1000,
        },  # Standard regularization
    ),
    SpectralClusteringModel: ExperimentConfig(
        model_class=SpectralClusteringModel,
        model_params={
            "n_clusters": 10,
            "random_state": 42,
            "affinity": "nearest_neighbors",
            "n_neighbors": 15,  # Scientific choice for digits
            "assign_labels": "kmeans",
            "gamma": 1.0,
        },
    ),
}


# Registry of experiment configurations for the wine dataset
WINE_EXPERIMENTS = {
    AgglomerativeClusteringModel: ExperimentConfig(
        model_class=AgglomerativeClusteringModel,
        model_params={"n_clusters": 3, "linkage": "ward"},  # 3 wine classes - no random_state
    ),
    FastICAModel: ExperimentConfig(
        model_class=FastICAModel,
        model_params={
            "n_components": 8,
            "random_state": 42,
            "max_iter": 1000,
            "tol": 1e-4,
        },  # Suitable for wine features
    ),
    GaussianMixtureModel: ExperimentConfig(
        model_class=GaussianMixtureModel,
        model_params={
            "n_components": 3,
            "random_state": 42,
            "covariance_type": "full",
            "max_iter": 100,
        },  # 3 wine classes
    ),
    GradientBoostingModel: ExperimentConfig(
        model_class=GradientBoostingModel,
        model_params={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
        },  # Conservative for small dataset
    ),
    MiniBatchKMeansModel: ExperimentConfig(
        model_class=MiniBatchKMeansModel,
        model_params={
            "n_clusters": 3,
            "random_state": 42,
            "batch_size": 50,
            "max_iter": 100,
        },  # Small batch for small dataset
    ),
    MLPModel: ExperimentConfig(
        model_class=MLPModel,
        model_params={
            "hidden_layer_sizes": (50, 25),
            "random_state": 42,
            "max_iter": 1000,
            "alpha": 0.01,
        },  # Conservative for small dataset
    ),
    RandomForestClassifierModel: ExperimentConfig(
        model_class=RandomForestClassifierModel,
        model_params={
            "n_estimators": 100,
            "max_depth": 8,
            "random_state": 42,
            "min_samples_split": 2,
        },  # Balanced for wine dataset
    ),
    RandomForestRegressorModel: ExperimentConfig(
        model_class=RandomForestRegressorModel,
        model_params={
            "n_estimators": 100,
            "max_depth": 8,
            "random_state": 42,
            "min_samples_split": 2,
        },
    ),
    RidgeModel: ExperimentConfig(
        model_class=RidgeModel,
        model_params={
            "alpha": 1.0,
            "random_state": 42,
            "max_iter": 1000,
        },  # Standard regularization
    ),
    SpectralClusteringModel: ExperimentConfig(
        model_class=SpectralClusteringModel,
        model_params={
            "n_clusters": 3,
            "random_state": 42,
            "affinity": "rbf",  # Better for dense continuous data
            "gamma": 1.0,  # Balanced gamma for wine data
            "assign_labels": "kmeans",  # More stable than discretize
        },
    ),
}


# Registry of experiment configurations for the LFW People dataset (face recognition)
LFW_PEOPLE_EXPERIMENTS = {
    RandomForestClassifierModel: ExperimentConfig(
        model_class=RandomForestClassifierModel,
        model_params={
            "n_estimators": 100,
            "max_depth": 15,
            "random_state": 42,
            "min_samples_split": 5,
        },  # Deep for face features
    ),
    MLPModel: ExperimentConfig(
        model_class=MLPModel,
        model_params={
            "hidden_layer_sizes": (200, 100),
            "random_state": 42,
            "max_iter": 500,
            "alpha": 0.01,
        },  # Deep for face recognition
    ),
    RidgeModel: ExperimentConfig(
        model_class=RidgeModel,
        model_params={
            "alpha": 10.0,
            "random_state": 42,
            "max_iter": 2000,
        },  # Higher regularization for high-dim
    ),
    FastICAModel: ExperimentConfig(
        model_class=FastICAModel,
        model_params={
            "n_components": 50,
            "random_state": 42,
            "max_iter": 1000,
            "tol": 1e-4,
        },  # Dimensionality reduction for faces
    ),
    GradientBoostingModel: ExperimentConfig(
        model_class=GradientBoostingModel,
        model_params={
            "n_estimators": 300,  # Significantly increased for 30-40s target
            "learning_rate": 0.03,  # Lower learning rate for better convergence
            "max_depth": 5,  # Deeper trees for complex face features
            "random_state": 42,
            "subsample": 0.9,  # High subsample for robust learning
            "max_features": "sqrt",  # Feature subsampling
            "validation_fraction": 0.15,  # Larger validation set
            "n_iter_no_change": 15,  # More patience for early stopping
            "tol": 1e-5,  # Stricter convergence tolerance
            "min_samples_split": 2,  # Standard splitting
            "min_samples_leaf": 1,  # Standard leaf size
        },  # Robust scientific parameters for face recognition (30-40 seconds target)
    ),
    RandomForestRegressorModel: ExperimentConfig(
        model_class=RandomForestRegressorModel,
        model_params={
            "n_estimators": 100,
            "max_depth": 15,
            "random_state": 42,
            "min_samples_split": 5,
        },
    ),
    # Clustering for face analysis
    AgglomerativeClusteringModel: ExperimentConfig(
        model_class=AgglomerativeClusteringModel,
        model_params={
            "n_clusters": 5,
            "linkage": "ward",
        },  # Multiple people clusters - no random_state
    ),
    MiniBatchKMeansModel: ExperimentConfig(
        model_class=MiniBatchKMeansModel,
        model_params={
            "n_clusters": 5,
            "random_state": 42,
            "batch_size": 256,
            "max_iter": 100,
        },  # Face clustering
    ),
    GaussianMixtureModel: ExperimentConfig(
        model_class=GaussianMixtureModel,
        model_params={
            "n_components": 5,
            "random_state": 42,
            "covariance_type": "diag",
            "max_iter": 100,
        },  # Diagonal for high-dim
    ),
    SpectralClusteringModel: ExperimentConfig(
        model_class=SpectralClusteringModel,
        model_params={
            "n_clusters": 5,
            "random_state": 42,
            "affinity": "nearest_neighbors",
            "n_neighbors": 10,
            "assign_labels": "kmeans",  # More stable for face data
        },
    ),
}


# Registry of experiment configurations for the Make Moons dataset (2D synthetic data)
MAKE_MOONS_EXPERIMENTS = {
    SpectralClusteringModel: ExperimentConfig(
        model_class=SpectralClusteringModel,
        model_params={
            "n_clusters": 2,  # Two moons
            "random_state": 42,
            "affinity": "nearest_neighbors",
            "n_neighbors": 15,  # Good for moon shapes
            "assign_labels": "kmeans",
            "gamma": 1.0,
        },
    ),
    AgglomerativeClusteringModel: ExperimentConfig(
        model_class=AgglomerativeClusteringModel,
        model_params={"n_clusters": 2, "linkage": "ward"},  # Two moons - no random_state
    ),
    MiniBatchKMeansModel: ExperimentConfig(
        model_class=MiniBatchKMeansModel,
        model_params={
            "n_clusters": 2,
            "random_state": 42,
            "batch_size": 100,
            "max_iter": 100,
        },  # Two moons
    ),
    GaussianMixtureModel: ExperimentConfig(
        model_class=GaussianMixtureModel,
        model_params={
            "n_components": 2,
            "random_state": 42,
            "covariance_type": "full",
            "max_iter": 100,
        },  # Two moons
    ),
    # Classification algorithms to compare with clustering
    RandomForestClassifierModel: ExperimentConfig(
        model_class=RandomForestClassifierModel,
        model_params={
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
            "min_samples_split": 5,
        },  # Simple for 2D data
    ),
    MLPModel: ExperimentConfig(
        model_class=MLPModel,
        model_params={
            "hidden_layer_sizes": (50, 25),
            "random_state": 42,
            "max_iter": 1000,
            "alpha": 0.01,
        },  # Simple for 2D
    ),
    GradientBoostingModel: ExperimentConfig(
        model_class=GradientBoostingModel,
        model_params={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
        },  # Standard for 2D
    ),
    RidgeModel: ExperimentConfig(
        model_class=RidgeModel,
        model_params={
            "alpha": 1.0,
            "random_state": 42,
            "max_iter": 1000,
        },  # Simple regularization for 2D
    ),
    FastICAModel: ExperimentConfig(
        model_class=FastICAModel,
        model_params={
            "n_components": 8,
            "random_state": 42,
            "max_iter": 1000,
            "tol": 1e-4,
        },  # Dimensionality reduction for 2D
    ),
}


# Registry of experiment configurations for the 20 Newsgroups dataset
NEWSGROUPS_20_EXPERIMENTS = {
    AgglomerativeClusteringModel: ExperimentConfig(
        model_class=AgglomerativeClusteringModel,
        model_params={
            "n_clusters": 8,
            "linkage": "ward",
        },  # Balanced clusters for text topics - no random_state
        dataset_name=SklearnDatasetName.NEWSGROUPS_20,
    ),
    FastICAModel: ExperimentConfig(
        model_class=FastICAModel,
        model_params={
            "n_components": 50,
            "random_state": 42,
            "max_iter": 1000,
            "tol": 1e-4,
        },  # Moderate components for text
        dataset_name=SklearnDatasetName.NEWSGROUPS_20,
    ),
    GaussianMixtureModel: ExperimentConfig(
        model_class=GaussianMixtureModel,
        model_params={
            "n_components": 8,
            "random_state": 42,
            "covariance_type": "diag",
            "max_iter": 100,
        },  # Diagonal for high-dim text
        dataset_name=SklearnDatasetName.NEWSGROUPS_20,
    ),
    GradientBoostingModel: ExperimentConfig(
        model_class=GradientBoostingModel,
        model_params={
            "n_estimators": 10,  # Muito reduzido para texto de alta dimensionalidade
            "learning_rate": 0.3,  # Maior para compensar menos estimadores
            "max_depth": 2,  # Muito raso para evitar overfitting em texto
            "random_state": 42,
            "subsample": 0.5,  # Agressivo subsampling para velocidade
            "max_features": "sqrt",  # Reduzir features para velocidade
        },  # Ultra-otimizado para texto de alta dimensionalidade
        dataset_name=SklearnDatasetName.NEWSGROUPS_20,
    ),
    MiniBatchKMeansModel: ExperimentConfig(
        model_class=MiniBatchKMeansModel,
        model_params={
            "n_clusters": 8,
            "random_state": 42,
            "batch_size": 512,
            "max_iter": 100,
        },  # Larger batch for text
        dataset_name=SklearnDatasetName.NEWSGROUPS_20,
    ),
    MLPModel: ExperimentConfig(
        model_class=MLPModel,
        model_params={
            "hidden_layer_sizes": (100, 50),
            "random_state": 42,
            "max_iter": 200,
            "alpha": 0.01,
        },  # Deep for text classification
        dataset_name=SklearnDatasetName.NEWSGROUPS_20,
    ),
    RandomForestClassifierModel: ExperimentConfig(
        model_class=RandomForestClassifierModel,
        model_params={
            "n_estimators": 100,
            "max_depth": 15,
            "random_state": 42,
            "min_samples_split": 5,
        },  # Deep for text features
        dataset_name=SklearnDatasetName.NEWSGROUPS_20,
    ),
    RandomForestRegressorModel: ExperimentConfig(
        model_class=RandomForestRegressorModel,
        model_params={
            "n_estimators": 100,
            "max_depth": 15,
            "random_state": 42,
            "min_samples_split": 5,
        },
        dataset_name=SklearnDatasetName.NEWSGROUPS_20,
    ),
    RidgeModel: ExperimentConfig(
        model_class=RidgeModel,
        model_params={
            "alpha": 10.0,
            "random_state": 42,
            "max_iter": 2000,
        },  # Higher regularization for text
        dataset_name=SklearnDatasetName.NEWSGROUPS_20,
    ),
    SpectralClusteringModel: ExperimentConfig(
        model_class=SpectralClusteringModel,
        model_params={
            "n_clusters": 8,  # Balanced for text topics
            "random_state": 42,
            "affinity": "rbf",  # Suitable for dense text features
            "gamma": 0.1,  # Conservative gamma for text
            "assign_labels": "kmeans",  # Stable assignment for text
        },
        dataset_name=SklearnDatasetName.NEWSGROUPS_20,
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
    elif dataset_name == SklearnDatasetName.LFW_PEOPLE:
        registry = LFW_PEOPLE_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.MAKE_MOONS:
        registry = MAKE_MOONS_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.NEWSGROUPS_20:
        registry = NEWSGROUPS_20_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.DIGITS:
        registry = DIGITS_EXPERIMENTS
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

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
    elif dataset_name == SklearnDatasetName.LFW_PEOPLE:
        registry = LFW_PEOPLE_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.MAKE_MOONS:
        registry = MAKE_MOONS_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.NEWSGROUPS_20:
        registry = NEWSGROUPS_20_EXPERIMENTS
    elif dataset_name == SklearnDatasetName.DIGITS:
        registry = DIGITS_EXPERIMENTS
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

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
