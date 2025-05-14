import numpy as np
from inference.transformations.data.base import InferenceTransformation

class ZeroOut(InferenceTransformation):
    def __init__(self, frac):
        self.frac = frac

    def apply(self, X):
        mask = np.random.rand(*X.shape) < self.frac
        X[mask] = 0.0
        return X

class InsertNaN(InferenceTransformation):
    def __init__(self, frac):
        self.frac = frac

    def apply(self, X):
        n_samples, n_features = X.shape
        n_missing = int(n_samples * n_features * self.frac)
        indices = np.unravel_index(np.random.choice(X.size, n_missing, replace=False), X.shape)
        X[indices] = np.nan
        return X
