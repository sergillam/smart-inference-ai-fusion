from inference.transformations.data.noise import GaussianNoise, FeatureSelectiveNoise
from inference.transformations.data.precision import TruncateDecimals, CastToInt, Quantize
from inference.transformations.data.structure import ShuffleFeatures, ScaleFeatures, RemoveFeatures, FeatureSwap
from inference.transformations.data.corruption import ZeroOut, InsertNaN
from inference.transformations.data.outliers import InjectOutliers
from inference.transformations.data.distraction import AddDummyFeatures, DuplicateFeatures
from utils.types import DatasetNoiseConfig

class InferenceEngine:
    def __init__(self, config: DatasetNoiseConfig):
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
        }

        for key, constructor in mapping.items():
            if key in config and config[key] is not None:
                transform = constructor(config[key])
                if transform:
                    self.pipeline.append(transform)

    def apply(self, X_train, X_test):
        for transform in self.pipeline:
            X_train = transform.apply(X_train)
            X_test = transform.apply(X_test)
        return X_train, X_test
