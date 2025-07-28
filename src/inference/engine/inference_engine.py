"""Inference engine for applying perturbations to input data (X).

Defines a pipeline that applies multiple data perturbation techniques based on DataNoiseConfig.
"""

from sklearn.impute import SimpleImputer
from inference.transformations.data.noise import GaussianNoise, FeatureSelectiveNoise
from inference.transformations.data.precision import TruncateDecimals, CastToInt, Quantize
from inference.transformations.data.structure import (
    ShuffleFeatures, ScaleFeatures, RemoveFeatures, FeatureSwap
)
from inference.transformations.data.corruption import ZeroOut, InsertNaN
from inference.transformations.data.outliers import InjectOutliers
from inference.transformations.data.distraction import AddDummyFeatures, DuplicateFeatures
from inference.transformations.data.conditional_noise import ConditionalNoise
from inference.transformations.data.cluster_swap import ClusterSwap
from inference.transformations.data.random_missing_block import RandomMissingBlock
from inference.transformations.data.distribution_shift import DistributionShiftMixing
from inference.transformations.data.group_outlier_injection import GroupOutlierInjection
from inference.transformations.data.temporal_drift_injection import TemporalDriftInjection

from utils.types import DataNoiseConfig

class InferenceEngine:
    """Pipeline for applying multiple perturbation techniques to input features (X)."""

    def __init__(self, config: DataNoiseConfig):
        """Initializes the inference pipeline with the provided configuration.

        Args:
            config (DataNoiseConfig): Configuration object specifying which perturbations to apply.
        """
        self.pipeline = self._build_pipeline(config)

    @staticmethod
    def _build_pipeline(config: DataNoiseConfig):
        """Builds the pipeline of transformations based on the configuration.

        Args:
            config (DataNoiseConfig): Configuration object.

        Returns:
            list: List of transformation instances.
        """
        # Mapping now uses direct class references where possible, removing unnecessary lambdas.
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
            # This long line is now broken for readability
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
        # The line that previously had trailing whitespace is now clean.
        return pipeline

    def apply(self, X_train, X_test):
        """Applies the sequence of transformations to X_train and X_test.

        Automatically imputes missing values before transformations that use clustering.

        Args:
            X_train (array-like): Training features.
            X_test (array-like): Test features.

        Returns:
            tuple: (X_train_perturbed, X_test_perturbed)
        """
        clustering_transforms = (ClusterSwap, GroupOutlierInjection)
        for transform in self.pipeline:
            if isinstance(transform, clustering_transforms):
                imputer = SimpleImputer(strategy='mean')
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)
            X_train = transform.apply(X_train)
            X_test = transform.apply(X_test)
        return X_train, X_test
