"""Inference engine for applying perturbations to input data (X).

This module defines the `InferenceEngine` class, responsible for dynamically
building and applying a configurable pipeline of data perturbation techniques.
These transformations simulate real-world noise, distribution shifts, outliers,
and other conditions to evaluate model robustness.

Example:
    >>> from smart_inference_ai_fusion.inference.engine.inference_engine import InferenceEngine
    >>> from smart_inference_ai_fusion.utils.types import DataNoiseConfig
    >>> config = DataNoiseConfig(noise_level=0.1)
    >>> engine = InferenceEngine(config)
    >>> X_train_perturbed, X_test_perturbed = engine.apply(X_train, X_test)
"""

from typing import Tuple

import numpy as np
from sklearn.impute import SimpleImputer

from smart_inference_ai_fusion.inference.transformations.data.cluster_swap import ClusterSwap
from smart_inference_ai_fusion.inference.transformations.data.conditional_noise import (
    ConditionalNoise,
)
from smart_inference_ai_fusion.inference.transformations.data.corruption import InsertNaN, ZeroOut
from smart_inference_ai_fusion.inference.transformations.data.distraction import (
    AddDummyFeatures,
    DuplicateFeatures,
)
from smart_inference_ai_fusion.inference.transformations.data.distribution_shift import (
    DistributionShiftMixing,
)
from smart_inference_ai_fusion.inference.transformations.data.group_outlier_injection import (
    GroupOutlierInjection,
)
from smart_inference_ai_fusion.inference.transformations.data.noise import (
    FeatureSelectiveNoise,
    GaussianNoise,
)
from smart_inference_ai_fusion.inference.transformations.data.outliers import InjectOutliers
from smart_inference_ai_fusion.inference.transformations.data.precision import (
    CastToInt,
    Quantize,
    TruncateDecimals,
)
from smart_inference_ai_fusion.inference.transformations.data.random_missing_block import (
    RandomMissingBlock,
)
from smart_inference_ai_fusion.inference.transformations.data.structure import (
    FeatureSwap,
    RemoveFeatures,
    ScaleFeatures,
    ShuffleFeatures,
)
from smart_inference_ai_fusion.inference.transformations.data.temporal_drift_injection import (
    TemporalDriftInjection,
)
from smart_inference_ai_fusion.utils.types import DataNoiseConfig


def _compute_data_statistics(X: np.ndarray, name: str) -> dict:
    """Compute statistics for a data array.

    Args:
        X: Data array to analyze
        name: Name identifier for the statistics

    Returns:
        Dictionary with statistics
    """
    # Guard against empty arrays to avoid ValueError from nanmin/nanmax
    if X.size == 0:
        return {
            f"{name}_shape": list(X.shape),
            f"{name}_mean": float("nan"),
            f"{name}_std": float("nan"),
            f"{name}_min": float("nan"),
            f"{name}_max": float("nan"),
            f"{name}_nan_count": 0,
            f"{name}_nan_fraction": 0.0,
        }

    stats = {
        f"{name}_shape": list(X.shape),
        f"{name}_mean": float(np.nanmean(X)),
        f"{name}_std": float(np.nanstd(X)),
        f"{name}_min": float(np.nanmin(X)),
        f"{name}_max": float(np.nanmax(X)),
        f"{name}_nan_count": int(np.isnan(X).sum()),
        f"{name}_nan_fraction": float(np.isnan(X).sum() / X.size),
    }
    return stats


def _compute_perturbation_diff(X_before: np.ndarray, X_after: np.ndarray) -> dict:
    """Compute perturbation statistics between original and transformed data.

    Args:
        X_before: Original data array
        X_after: Transformed data array

    Returns:
        Dictionary with perturbation statistics
    """
    # Handle shape differences (features may have been added/removed)
    if X_before.shape != X_after.shape:
        return {
            "shape_changed": True,
            "original_shape": list(X_before.shape),
            "transformed_shape": list(X_after.shape),
            "features_added": (
                max(0, X_after.shape[1] - X_before.shape[1]) if len(X_after.shape) > 1 else 0
            ),
            "features_removed": (
                max(0, X_before.shape[1] - X_after.shape[1]) if len(X_before.shape) > 1 else 0
            ),
        }

    # Compute element-wise differences
    diff = X_after - X_before
    changed_mask = ~np.isclose(X_before, X_after, equal_nan=True)

    stats = {
        "shape_changed": False,
        "samples_affected": (
            int(np.any(changed_mask, axis=1).sum())
            if len(changed_mask.shape) > 1
            else int(changed_mask.sum())
        ),
        "samples_affected_fraction": (
            float(np.any(changed_mask, axis=1).mean())
            if len(changed_mask.shape) > 1
            else float(changed_mask.mean())
        ),
        "elements_changed": int(changed_mask.sum()),
        "elements_changed_fraction": float(changed_mask.mean()),
        "mean_absolute_change": float(np.nanmean(np.abs(diff))),
        "max_absolute_change": float(np.nanmax(np.abs(diff))) if changed_mask.any() else 0.0,
        "mean_relative_change": float(np.nanmean(np.abs(diff) / (np.abs(X_before) + 1e-10))),
    }
    return stats


class InferenceEngine:
    """Pipeline for applying multiple perturbation techniques to feature data (X).

    This engine dynamically constructs a sequence of transformations based
    on a configuration object (`DataNoiseConfig`). Each transformation
    simulates a type of data imperfection to test model robustness.
    """

    def __init__(self, config: DataNoiseConfig):
        """Initialize the inference engine with a specific configuration.

        Args:
            config (DataNoiseConfig): Configuration object specifying which
                perturbations to include in the pipeline.
        """
        self.pipeline = self._build_pipeline(config)

    @staticmethod
    def _build_pipeline(config: DataNoiseConfig):
        """Build the transformation pipeline from the given configuration.

        Args:
            config (DataNoiseConfig): Configuration object with transformation parameters.

        Returns:
            list: A list of instantiated transformations, in the order they should be applied.

        Raises:
            AttributeError: If the configuration references an unknown parameter key.
        """
        factory_mapping = {
            "noise_level": GaussianNoise,
            "feature_selective_noise": lambda val: FeatureSelectiveNoise(*val),
            "truncate_decimals": TruncateDecimals,
            "cast_to_int": lambda val: CastToInt() if val else None,
            "quantize_bins": Quantize,
            "shuffle_fraction": ShuffleFeatures,
            "scale_range": ScaleFeatures,
            "zero_out_fraction": ZeroOut,
            "insert_nan_fraction": InsertNaN,
            "outlier_fraction": InjectOutliers,
            "add_dummy_features": AddDummyFeatures,
            "duplicate_features": DuplicateFeatures,
            "remove_features": RemoveFeatures,
            "feature_swap": FeatureSwap,
            "conditional_noise": lambda val: ConditionalNoise(val[0], val[1]),
            "cluster_swap_fraction": lambda val: ClusterSwap(swap_fraction=val),
            "random_missing_block_fraction": lambda val: RandomMissingBlock(block_fraction=val),
            "distribution_shift_fraction": lambda val: DistributionShiftMixing(shift_fraction=val),
            "group_outlier_cluster_fraction": (
                lambda val: GroupOutlierInjection(outlier_fraction=val)
            ),
            "temporal_drift_std": lambda val: TemporalDriftInjection(drift_std=val),
        }

        pipeline = []
        for key, factory in factory_mapping.items():
            value = getattr(config, key, None)
            if value is not None:
                transform_obj = factory(value)
                if transform_obj is not None:
                    pipeline.append(transform_obj)

        return pipeline

    def apply(
        self, X_train: np.ndarray, X_test: np.ndarray, collect_statistics: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Apply the pipeline of transformations to training and test data.

        Automatically imputes missing values (with `mean` strategy) before
        transformations that require complete data, such as clustering-based
        perturbations.

        Args:
            X_train (np.ndarray): Training feature matrix.
                Shape: (n_samples_train, n_features).
            X_test (np.ndarray): Test feature matrix.
                Shape: (n_samples_test, n_features).
            collect_statistics (bool): Whether to collect transformation statistics.
                Defaults to False. When True, samples up to 10,000 rows for
                per-transformation statistics to limit memory overhead.

        Returns:
            Tuple[np.ndarray, np.ndarray, dict]: A tuple `(X_train_perturbed, X_test_perturbed, statistics)`
            after applying the pipeline. The statistics dict contains transformation details.

        Raises:
            ValueError: If `X_train` and `X_test` have mismatched feature dimensions.
            RuntimeError: If any transformation fails during execution.
        """
        # Store original data for statistics
        X_train_original = X_train.copy() if collect_statistics else None
        X_test_original = X_test.copy() if collect_statistics else None

        statistics = {
            "transformations_applied": [],
            "per_transformation_stats": [],
        }

        if collect_statistics:
            statistics["original"] = {
                "train": _compute_data_statistics(X_train, "train"),
                "test": _compute_data_statistics(X_test, "test"),
            }

        clustering_transforms = (ClusterSwap, GroupOutlierInjection)

        # Maximum sample size for per-transformation statistics (reduces memory overhead)
        _STATS_SAMPLE_SIZE = 10_000

        for transform in self.pipeline:
            transform_name = transform.__class__.__name__

            # Store sampled state before transformation for per-transform stats
            if collect_statistics:
                n_samples = X_train.shape[0]
                if n_samples > _STATS_SAMPLE_SIZE:
                    sample_idx = np.random.choice(n_samples, _STATS_SAMPLE_SIZE, replace=False)
                    X_train_sample_before = X_train[sample_idx].copy()
                else:
                    sample_idx = None
                    X_train_sample_before = X_train.copy()
            else:
                sample_idx = None
                X_train_sample_before = None

            if isinstance(transform, clustering_transforms):
                imputer = SimpleImputer(strategy="mean")
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)
            X_train = transform.apply(X_train)
            X_test = transform.apply(X_test)

            # Track transformation statistics
            if collect_statistics:
                statistics["transformations_applied"].append(transform_name)
                if X_train_sample_before is not None:
                    # Use same sample indices for after comparison
                    if sample_idx is not None:
                        X_train_sample_after = X_train[sample_idx]
                    else:
                        X_train_sample_after = X_train
                    transform_stats = _compute_perturbation_diff(
                        X_train_sample_before, X_train_sample_after
                    )
                    transform_stats["transformation_name"] = transform_name
                    transform_stats["sampled"] = sample_idx is not None
                    statistics["per_transformation_stats"].append(transform_stats)

        if collect_statistics:
            statistics["transformed"] = {
                "train": _compute_data_statistics(X_train, "train"),
                "test": _compute_data_statistics(X_test, "test"),
            }
            # Overall perturbation summary
            statistics["overall_perturbation"] = {
                "train": _compute_perturbation_diff(X_train_original, X_train),
                "test": _compute_perturbation_diff(X_test_original, X_test),
            }

        return X_train, X_test, statistics
