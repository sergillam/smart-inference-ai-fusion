import numpy as np
from inference.transformations.data.base import InferenceTransformation

class GaussianNoise(InferenceTransformation):
    def __init__(self, level):
        self.level = level

    def apply(self, X):
        return X + np.random.normal(0, self.level, X.shape)

class FeatureSelectiveNoise(InferenceTransformation):
    def __init__(self, level, features):
        self.level = level
        self.features = features
    def apply(self, X):
        X_new = X.copy()
        for f in self.features:
            X_new[:, f] += np.random.normal(0, self.level, X.shape[0])
        return X_new