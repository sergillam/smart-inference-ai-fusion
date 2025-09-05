"""Common utilities for experiment scripts."""

import time
from typing import Callable, Dict, Optional, Type, Union

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.core.experiment import Experiment
from smart_inference_ai_fusion.datasets.factory import DatasetFactory
from smart_inference_ai_fusion.inference.pipeline.inference_pipeline import InferencePipeline
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
from smart_inference_ai_fusion.utils.preprocessing import (
    filter_sklearn_params,
    validate_sklearn_params,
)
from smart_inference_ai_fusion.utils.report import (
    ReportMode,
    generate_experiment_filename,
    report_data,
)
from smart_inference_ai_fusion.utils.types import (
    DataNoiseConfig,
    DatasetSourceType,
    LabelNoiseConfig,
    ParameterNoiseConfig,
    SklearnDatasetName,
)


def _create_classification_label_config() -> LabelNoiseConfig:
    """Create custom label configuration for classification models.

    Removes flip_near_border_fraction to avoid convergence issues.
    """
    return LabelNoiseConfig(
        confusion_matrix_noise_level=0.1,
        partial_label_fraction=0.1,
        swap_within_class_fraction=0.1,
    )


def _create_random_forest_label_config() -> LabelNoiseConfig:
    """Create custom label configuration for RandomForestClassifier.

    Removes flip_near_border_fraction to avoid convergence issues.
    """
    return _create_classification_label_config()


def _create_mlp_label_config() -> LabelNoiseConfig:
    """Create custom label configuration for MLPModel.

    Uses same configuration as other classification models.
    """
    return _create_classification_label_config()


def _create_gradient_boosting_label_config() -> LabelNoiseConfig:
    """Create custom label configuration for GradientBoostingModel.

    Uses same configuration as other classification models.
    """
    return _create_classification_label_config()


def _create_random_forest_regressor_label_config() -> LabelNoiseConfig:
    """Create custom label configuration for RandomForestRegressor.

    Excludes flip_near_border_fraction for regression models (no predict_proba).
    """
    return LabelNoiseConfig(
        label_noise_fraction=0.1,
        confusion_matrix_noise_level=0.1,
        partial_label_fraction=0.1,
        swap_within_class_fraction=0.1,
    )


def _create_regression_label_config() -> LabelNoiseConfig:
    """Create custom label configuration for regression models.

    Excludes flip_near_border_fraction (requires predict_proba, not available in regression).
    """
    return LabelNoiseConfig(
        label_noise_fraction=0.1,
        confusion_matrix_noise_level=0.1,
        partial_label_fraction=0.1,
        swap_within_class_fraction=0.1,
    )


def _create_clustering_label_config() -> LabelNoiseConfig:
    """Create custom label configuration for clustering models.

    Clustering models don't use supervised labels in the same way, so we exclude
    transformations that require supervised model interfaces.
    """
    return LabelNoiseConfig(
        # Only include basic label transformations that work with clustering
        label_noise_fraction=0.1,
        # Exclude transformations that require supervised model methods
        # confusion_matrix_noise_level=0.1,  # Requires class predictions
        # partial_label_fraction=0.1,        # Requires supervised context
        # swap_within_class_fraction=0.1,    # Requires class structure
    )


# Registry of model-specific configuration overrides
MODEL_CONFIG_OVERRIDES: Dict[Type[BaseModel], Dict[str, Callable]] = {
    # Classification models with special needs
    RandomForestClassifierModel: {
        "label_config": _create_random_forest_label_config,
    },
    MLPModel: {
        "label_config": _create_mlp_label_config,
    },
    GradientBoostingModel: {
        "label_config": _create_gradient_boosting_label_config,
    },
    # Regression models (no predict_proba)
    RandomForestRegressorModel: {
        "label_config": _create_random_forest_regressor_label_config,
    },
    RidgeModel: {
        "label_config": _create_regression_label_config,
    },
    # Clustering models (unsupervised)
    MiniBatchKMeansModel: {
        "label_config": _create_clustering_label_config,
    },
    SpectralClusteringModel: {
        "label_config": _create_clustering_label_config,
    },
    AgglomerativeClusteringModel: {
        "label_config": _create_clustering_label_config,
    },
    GaussianMixtureModel: {
        "label_config": _create_clustering_label_config,
    },
    FastICAModel: {
        "label_config": _create_clustering_label_config,
    },
}


def get_model_specific_configs(model_class: Type[BaseModel], model_name: str) -> Dict[str, any]:
    """Get model-specific configuration overrides.

    Args:
        model_class: The model class to get configurations for
        model_name: Model name for logging purposes

    Returns:
        Dict containing any custom configurations for the model
    """
    configs = {}

    if model_class in MODEL_CONFIG_OVERRIDES:
        overrides = MODEL_CONFIG_OVERRIDES[model_class]

        for config_type, config_factory in overrides.items():
            configs[config_type] = config_factory()

            # Extract the first line of docstring for cleaner logging
            docstring = config_factory.__doc__
            if docstring:
                first_line = docstring.strip().split("\n")[0]
                doc_summary = first_line
            else:
                doc_summary = "custom configuration"

            report_data(
                f"INFO: Applying custom {config_type} for {model_name}: {doc_summary}",
                mode=ReportMode.PRINT,
            )

    return configs


def create_dataset(source_type: DatasetSourceType, dataset_name: Union[SklearnDatasetName, str]):
    """Create a dataset from various sources.

    Args:
        source_type: Type of dataset source (SKLEARN, CSV, etc.)
        dataset_name: Name/identifier of the dataset

    Returns:
        Dataset instance ready for experiments

    Examples:
        >>> create_dataset(DatasetSourceType.SKLEARN, SklearnDatasetName.DIGITS)
        >>> create_dataset(DatasetSourceType.SKLEARN, SklearnDatasetName.IRIS)
        >>> create_dataset(DatasetSourceType.CSV, "titanic")
    """
    if source_type == DatasetSourceType.SKLEARN:
        # Handle both enum and string inputs for sklearn datasets
        if isinstance(dataset_name, SklearnDatasetName):
            name = dataset_name
        else:
            # Convert string to enum if needed
            name = SklearnDatasetName(dataset_name)
        return DatasetFactory.create(DatasetSourceType.SKLEARN, name=name)

    if source_type == DatasetSourceType.CSV:
        # For CSV datasets, dataset_name should be a string path/identifier
        return DatasetFactory.create(DatasetSourceType.CSV, name=str(dataset_name))

    raise ValueError(f"Unsupported dataset source type: {source_type}")


def create_digits_dataset():
    """Create the standard digits dataset for experiments.

    Deprecated: Use create_dataset(DatasetSourceType.SKLEARN, SklearnDatasetName.DIGITS) instead.
    """
    return create_dataset(DatasetSourceType.SKLEARN, SklearnDatasetName.DIGITS)


def create_inference_configs():
    """Create standard inference configuration objects."""
    data_config = DataNoiseConfig(
        noise_level=0.2,
        truncate_decimals=1,
        quantize_bins=5,
        cast_to_int=False,
        shuffle_fraction=0.1,
        scale_range=(0.8, 1.2),
        zero_out_fraction=0.05,
        insert_nan_fraction=0.05,
        outlier_fraction=0.05,
        add_dummy_features=2,
        duplicate_features=2,
        feature_selective_noise=(0.3, [0, 2]),
        remove_features=[1, 3],
        feature_swap=[0, 2],
        conditional_noise=(0, 5.0, 0.2),
        random_missing_block_fraction=0.1,
        distribution_shift_fraction=0.1,
        cluster_swap_fraction=0.1,
        group_outlier_cluster_fraction=0.1,
        temporal_drift_std=0.5,
    )
    label_config = LabelNoiseConfig(
        label_noise_fraction=0.1,
        flip_near_border_fraction=0.1,
        confusion_matrix_noise_level=0.1,
        partial_label_fraction=0.1,
        swap_within_class_fraction=0.1,
    )
    param_config = ParameterNoiseConfig(
        integer_noise=True,
        boolean_flip=True,
        string_mutator=True,
        semantic_mutation=True,
        scale_hyper=True,
        cross_dependency=True,
        random_from_space=True,
        bounded_numeric=True,
        type_cast_perturbation=True,
        enum_boundary_shift=True,
    )

    return data_config, label_config, param_config


def create_inference_configs_lfw_people():
    """Create inference configuration objects specific for LFW People dataset.

    This configuration is optimized for high-dimensional image data from the
    LFW People dataset.
    """
    data_config = DataNoiseConfig(
        noise_level=0.1,  # Lower noise for high-dim data
        truncate_decimals=2,  # More precision for image data
        quantize_bins=10,  # More bins for image intensities
        cast_to_int=False,
        shuffle_fraction=0.05,  # Lower shuffle for stability
        scale_range=(0.9, 1.1),  # Smaller scale changes
        zero_out_fraction=0.02,  # Lower zero-out for images
        insert_nan_fraction=0.02,  # Lower NaN insertion
        outlier_fraction=0.02,  # Lower outlier injection
        add_dummy_features=10,  # More dummy features for high-dim
        duplicate_features=5,  # Some feature duplication
        feature_selective_noise=(0.1, [0, 1, 2, 3, 4]),  # Noise on first few features only
        remove_features=[],  # Don't remove features from high-dim data
        feature_swap=[0, 1],  # Minimal feature swapping
        conditional_noise=(0, 2.0, 0.1),  # Lower conditional noise
        random_missing_block_fraction=0.05,
        distribution_shift_fraction=0.05,
        cluster_swap_fraction=0.05,
        group_outlier_cluster_fraction=0.05,
        temporal_drift_std=0.2,  # Lower drift for stability
    )
    label_config = LabelNoiseConfig(
        label_noise_fraction=0.05,  # Lower label noise for face recognition
        flip_near_border_fraction=0.05,
        confusion_matrix_noise_level=0.05,
        partial_label_fraction=0.05,
        swap_within_class_fraction=0.05,
    )
    param_config = ParameterNoiseConfig(
        integer_noise=True,
        boolean_flip=True,
        string_mutator=True,
        semantic_mutation=True,
        scale_hyper=True,
        cross_dependency=True,
        random_from_space=True,
        bounded_numeric=True,
        type_cast_perturbation=True,
        enum_boundary_shift=True,
    )

    return data_config, label_config, param_config


def create_inference_configs_make_moons():
    """Create inference configuration objects specific for Make Moons dataset (2 features only)."""
    data_config = DataNoiseConfig(
        noise_level=0.2,
        truncate_decimals=1,
        quantize_bins=5,
        cast_to_int=False,
        shuffle_fraction=0.1,
        scale_range=(0.8, 1.2),
        zero_out_fraction=0.05,
        insert_nan_fraction=0.05,
        outlier_fraction=0.05,
        add_dummy_features=1,  # Reduced for 2-feature dataset
        duplicate_features=1,  # Reduced for 2-feature dataset
        feature_selective_noise=(0.3, [0]),  # Only use valid feature indices [0, 1]
        remove_features=[1],  # Only remove one feature, leaving [0]
        feature_swap=[0, 1],  # Swap only available features
        conditional_noise=(0, 5.0, 0.2),
        random_missing_block_fraction=0.1,
        distribution_shift_fraction=0.1,
        cluster_swap_fraction=0.1,
        group_outlier_cluster_fraction=0.1,
        temporal_drift_std=0.5,
    )
    label_config = LabelNoiseConfig(
        label_noise_fraction=0.1,
        flip_near_border_fraction=0.1,
        confusion_matrix_noise_level=0.1,
        partial_label_fraction=0.1,
        swap_within_class_fraction=0.1,
    )
    param_config = ParameterNoiseConfig(
        integer_noise=True,
        boolean_flip=True,
        string_mutator=True,
        semantic_mutation=True,
        scale_hyper=True,
        cross_dependency=True,
        random_from_space=True,
        bounded_numeric=True,
        type_cast_perturbation=True,
        enum_boundary_shift=True,
    )

    return data_config, label_config, param_config


def run_baseline_experiment(
    model_class: Type[BaseModel],
    model_name: str,
    dataset_source: DatasetSourceType,
    dataset_name: Union[SklearnDatasetName, str],
    filtered_params: Optional[dict] = None,
):
    """Run baseline experiment without inference.

    Args:
        model_class (Type[BaseModel]): Model class to instantiate.
        model_name (str): Name for logging and reporting.
        dataset_source (DatasetSourceType): Source type of the dataset (SKLEARN, CSV, etc.).
        dataset_name (SklearnDatasetName | str): Name/identifier of the dataset.
        filtered_params (Optional[dict]): Optional filtered parameters for the model.

    Returns:
        dict: Experiment results.
    """
    # Start timing
    start_time = time.time()

    report_data(f"=== {dataset_name} WITHOUT INFERENCE {model_name} ===", mode=ReportMode.PRINT)

    # Create dataset and model using generic function
    dataset = create_dataset(dataset_source, dataset_name)
    X_train, X_test, y_train, y_test = dataset.load_data()

    params = filtered_params or {}
    model = model_class(**params)

    # Run experiment
    experiment = Experiment(model=model, dataset=dataset)
    metrics = experiment.run(X_train, X_test, y_train, y_test)

    # Calculate execution time
    execution_time = time.time() - start_time

    report_data("Evaluation metrics (no inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(f"Execution time: {execution_time:.2f} seconds", mode=ReportMode.PRINT)

    # Add execution time to metrics
    metrics["execution_time_seconds"] = execution_time

    # Note: Individual JSON files no longer generated - results will be in consolidated format

    return metrics


def run_inference_experiment(
    model_class: Type[BaseModel],
    model_name: str,
    dataset_source: DatasetSourceType,
    dataset_name: Union[SklearnDatasetName, str],
    filtered_params: Optional[dict] = None,
):
    """Run experiment with full inference pipeline.

    Args:
        model_class (Type[BaseModel]): Model class to instantiate.
        model_name (str): Name for logging and reporting.
        dataset_source (DatasetSourceType): Source type of the dataset (SKLEARN, CSV, etc.).
        dataset_name (SklearnDatasetName | str): Name/identifier of the dataset.
        filtered_params (Optional[dict]): Optional filtered parameters for the model.

    Returns:
        dict: Experiment results.
    """
    # Start timing
    start_time = time.time()

    report_data(
        f"=== {dataset_name} WITH INFERENCE (data + param + label) {model_name} ===",
        mode=ReportMode.PRINT,
    )

    # Create base configurations - use dataset specific config if needed
    if (
        isinstance(dataset_name, SklearnDatasetName)
        and dataset_name == SklearnDatasetName.MAKE_MOONS
    ):
        data_config, label_config, _ = create_inference_configs_make_moons()
    elif (
        isinstance(dataset_name, SklearnDatasetName)
        and dataset_name == SklearnDatasetName.LFW_PEOPLE
    ):
        data_config, label_config, _ = create_inference_configs_lfw_people()
    else:
        data_config, label_config, _ = create_inference_configs()  # param_config is unused here

    # Apply model-specific configuration overrides
    model_configs = get_model_specific_configs(model_class, model_name)
    label_config = model_configs.get("label_config", label_config)

    # Create dataset and load data using generic function
    dataset = create_dataset(dataset_source, dataset_name)
    X_train, X_test, y_train, y_test = dataset.load_data()

    # Create pipeline and model
    pipeline = InferencePipeline(
        data_noise_config=data_config, label_noise_config=label_config, X_train=X_train
    )

    params = filtered_params or {}

    # Apply inference transformations
    X_train, X_test = pipeline.apply_data_inference(X_train, X_test)
    y_train, y_test = pipeline.apply_label_inference(
        y_train, y_test, model=None, X_train=X_train, X_test=X_test
    )

    # Apply parameter inference and create model
    model, _ = pipeline.apply_param_inference(model_class, params)  # param_info is unused

    # Validate and fix any invalid parameters that may have been introduced by inference
    if (
        hasattr(model, "model")
        and model.model is not None
        and hasattr(model.model, "get_params")
        and hasattr(model.model, "__class__")
    ):
        # Get the actual sklearn model parameters
        sklearn_params = model.model.get_params()
        validated_params = validate_sklearn_params(sklearn_params, model.model.__class__)
        # Update the model if any corrections were made
        if validated_params != sklearn_params:
            model.model.set_params(**validated_params)

    # Run experiment
    experiment = Experiment(model=model, dataset=dataset)
    metrics = experiment.run(X_train, X_test, y_train, y_test)

    # Calculate execution time
    execution_time = time.time() - start_time

    report_data("Evaluation metrics (with inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(f"Execution time: {execution_time:.2f} seconds", mode=ReportMode.PRINT)

    # Add execution time to metrics
    metrics["execution_time_seconds"] = execution_time

    # Note: Individual JSON files no longer generated - results will be in consolidated format

    return metrics


def run_standard_experiment(
    model_class: Type[BaseModel],
    model_name: str,
    dataset_source: DatasetSourceType,
    dataset_name: Union[SklearnDatasetName, str],
    model_params: Optional[dict] = None,
):
    """Run both baseline and inference experiments for a model on any dataset.

    Args:
        model_class (Type[BaseModel]): Model class to instantiate.
        model_name (str): Name for logging and reporting.
        dataset_source (DatasetSourceType): Source type of the dataset (SKLEARN, CSV, etc.).
        dataset_name (SklearnDatasetName | str): Name/identifier of the dataset.
        model_params (Optional[dict]): Optional parameters specific to the model.

    Returns:
        tuple: A tuple of (baseline_metrics, inference_metrics).

    Examples:
        >>> # Run on sklearn digits dataset
        >>> run_standard_experiment(
        ...     RandomForestClassifierModel,
        ...     "RandomForest",
        ...     DatasetSourceType.SKLEARN,
        ...     SklearnDatasetName.DIGITS
        ... )
        >>>
        >>> # Run on sklearn iris dataset
        >>> run_standard_experiment(
        ...     SVMModel,
        ...     "SVM",
        ...     DatasetSourceType.SKLEARN,
        ...     SklearnDatasetName.IRIS
        ... )
        >>>
        >>> # Run on custom CSV dataset
        >>> run_standard_experiment(
        ...     KNNModel,
        ...     "KNN",
        ...     DatasetSourceType.CSV,
        ...     "titanic"
        ... )
    """
    # Start timing for total experiment duration
    total_start_time = time.time()

    # Filter parameters if it's a scikit-learn compatible model wrapper
    filtered_params = model_params
    if model_params:
        try:
            # Check if the model is one of our wrappers with a .model attribute
            sample_model = model_class()
            if hasattr(sample_model, "model"):
                filtered_params = filter_sklearn_params(sample_model.model.__class__, model_params)
        except (TypeError, ValueError, AttributeError):
            # If instantiation or filtering fails, use the original params
            filtered_params = model_params

    # Run experiments
    baseline_metrics = run_baseline_experiment(
        model_class, model_name, dataset_source, dataset_name, filtered_params
    )
    inference_metrics = run_inference_experiment(
        model_class, model_name, dataset_source, dataset_name, filtered_params
    )

    # Calculate total execution time
    total_execution_time = time.time() - total_start_time

    # Report total execution summary
    report_data(
        f"\n=== EXPERIMENT SUMMARY FOR {model_name} on {dataset_name} ===", mode=ReportMode.PRINT
    )
    report_data(
        f"Baseline execution time: {baseline_metrics.get('execution_time_seconds', 0):.2f} seconds",
        mode=ReportMode.PRINT,
    )
    inference_time = inference_metrics.get("execution_time_seconds", 0)
    report_data(
        f"Inference execution time: {inference_time:.2f} seconds",
        mode=ReportMode.PRINT,
    )
    report_data(
        f"Total experiment time: {total_execution_time:.2f} seconds",
        mode=ReportMode.PRINT,
    )

    # Create comprehensive consolidated results
    baseline_time = baseline_metrics.get("execution_time_seconds", 0)
    inference_time = inference_metrics.get("execution_time_seconds", 0)

    consolidated_results = {
        "experiment_info": {
            "model_name": model_name,
            "model_class": model_class.__name__,
            "dataset_name": str(dataset_name),
            "dataset_source": (
                dataset_source.value if hasattr(dataset_source, "value") else str(dataset_source)
            ),
            "model_params": filtered_params or {},
            "timestamp": baseline_metrics.get("timestamp") or inference_metrics.get("timestamp"),
        },
        "results": {
            "without_inference": {
                "metrics": {
                    k: v for k, v in baseline_metrics.items() if k != "execution_time_seconds"
                },
                "execution_time_seconds": baseline_time,
            },
            "with_inference": {
                "metrics": {
                    k: v for k, v in inference_metrics.items() if k != "execution_time_seconds"
                },
                "execution_time_seconds": inference_time,
            },
        },
        "timing_analysis": {
            "baseline_time_seconds": baseline_time,
            "inference_time_seconds": inference_time,
            "total_experiment_time_seconds": total_execution_time,
            "inference_overhead_seconds": inference_time - baseline_time,
            "inference_overhead_percentage": (
                (inference_time - baseline_time) / max(baseline_time, 0.001)
            )
            * 100,
            "time_comparison": {
                "faster_approach": "baseline" if baseline_time < inference_time else "inference",
                "time_difference_seconds": abs(inference_time - baseline_time),
                "speed_ratio": max(inference_time, baseline_time)
                / max(min(inference_time, baseline_time), 0.001),
            },
        },
    }

    # Generate dataset identifier for filename
    dataset_id = (
        dataset_name.value if isinstance(dataset_name, SklearnDatasetName) else str(dataset_name)
    )

    # Save comprehensive consolidated results
    report_data(
        content=consolidated_results,
        name_output=generate_experiment_filename(model_class, dataset_id, "complete-results"),
        mode=ReportMode.JSON_RESULT,
    )

    return baseline_metrics, inference_metrics


# Backward compatibility function for existing digits experiments
def run_standard_experiment_digits(
    model_class: Type[BaseModel], model_name: str, model_params: Optional[dict] = None
):
    """Run standard experiment specifically for digits dataset.

    This is a convenience function that maintains backward compatibility
    with existing experiment scripts.

    Args:
        model_class (Type[BaseModel]): Model class to instantiate.
        model_name (str): Name for logging and reporting.
        model_params (Optional[dict]): Optional parameters specific to the model.

    Returns:
        tuple: A tuple of (baseline_metrics, inference_metrics).
    """
    return run_standard_experiment(
        model_class=model_class,
        model_name=model_name,
        dataset_source=DatasetSourceType.SKLEARN,
        dataset_name=SklearnDatasetName.DIGITS,
        model_params=model_params,
    )
