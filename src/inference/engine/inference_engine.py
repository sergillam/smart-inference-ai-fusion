from inference.transformations.data.noise import GaussianNoise, FeatureSelectiveNoise
from inference.transformations.data.precision import TruncateDecimals, CastToInt, Quantize
from inference.transformations.data.structure import ShuffleFeatures, ScaleFeatures, RemoveFeatures, FeatureSwap
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
    """
    Pipeline para aplicar múltiplas técnicas de perturbação nos dados de entrada (X).

    Configuração via DataNoiseConfig.
    """
    def __init__(self, config: DataNoiseConfig):
        self.pipeline = []

        mapping = {
            "noise_level": lambda val: GaussianNoise(val),
            "feature_selective_noise": lambda val: FeatureSelectiveNoise(*val),
            "truncate_decimals": lambda val: TruncateDecimals(val),
            "cast_to_int": lambda val: CastToInt() if val else None,
            "quantize_bins": lambda val: Quantize(val),
            "shuffle_fraction": lambda val: ShuffleFeatures(val),
            "scale_range": lambda val: ScaleFeatures(val),
            "zero_out_fraction": lambda val: ZeroOut(val),
            "insert_nan_fraction": lambda val: InsertNaN(val),
            "outlier_fraction": lambda val: InjectOutliers(val),
            "add_dummy_features": lambda val: AddDummyFeatures(val),
            "duplicate_features": lambda val: DuplicateFeatures(val),
            "remove_features": lambda val: RemoveFeatures(val),
            "feature_swap": lambda val: FeatureSwap(val),
            "conditional_noise": lambda val: ConditionalNoise(val[0], val[1]),
            "cluster_swap_fraction": lambda val: ClusterSwap(swap_fraction=val),
            "random_missing_block_fraction": lambda val: RandomMissingBlock(block_fraction=val),
            "distribution_shift_fraction": lambda val: DistributionShiftMixing(shift_fraction=val),
            "group_outlier_cluster_fraction": lambda val: GroupOutlierInjection(outlier_fraction=val),
            "temporal_drift_std": lambda val: TemporalDriftInjection(drift_std=val),
        }

        for key, constructor in mapping.items():
            if key in config and getattr(config, key) is not None:
                transform = constructor(getattr(config, key))
                if transform:
                    self.pipeline.append(transform)

    def apply(self, X_train, X_test):
        for transform in self.pipeline:
            X_train = transform.apply(X_train)
            X_test = transform.apply(X_test)
        return X_train, X_test
