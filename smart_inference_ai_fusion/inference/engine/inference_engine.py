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

    def apply(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the pipeline of transformations to training and test data.

        Automatically imputes missing values (with `mean` strategy) before
        transformations that require complete data, such as clustering-based
        perturbations.

        Args:
            X_train (np.ndarray): Training feature matrix.
                Shape: (n_samples_train, n_features).
            X_test (np.ndarray): Test feature matrix.
                Shape: (n_samples_test, n_features).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple `(X_train_perturbed, X_test_perturbed)`
            after applying the pipeline.

        Raises:
            ValueError: If `X_train` and `X_test` have mismatched feature dimensions.
            RuntimeError: If any transformation fails during execution.
        """
        clustering_transforms = (ClusterSwap, GroupOutlierInjection)
        for transform in self.pipeline:
            if isinstance(transform, clustering_transforms):
                imputer = SimpleImputer(strategy="mean")
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)
            X_train = transform.apply(X_train)
            X_test = transform.apply(X_test)
        return X_train, X_test
