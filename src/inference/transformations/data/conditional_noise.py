import numpy as np
from inference.transformations.data.base import InferenceTransformation

class ConditionalNoise(InferenceTransformation):
    """
    Applies Gaussian noise only to samples that meet a condition on a given feature.
    For example: apply noise if age > 60.
    """
    def __init__(self, feature_index: int, threshold: float, noise_std: float = 0.1):
        self.feature_index = feature_index
        self.threshold = threshold
        self.noise_std = noise_std

    def apply(self, X):
        X = np.array(X, dtype=np.float32)
        condition = X[:, self.feature_index] > self.threshold
        noise = np.random.normal(0, self.noise_std, size=X.shape)
        X[condition] += noise[condition]
        return X
