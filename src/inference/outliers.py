import numpy as np
from inference.base_inference import InferenceTransformation

class InjectOutliers(InferenceTransformation):
    def __init__(self, frac):
        self.frac = frac

    def apply(self, X):
        n_samples, n_features = X.shape
        n_outliers = int(n_samples * n_features * self.frac)
        indices = np.unravel_index(np.random.choice(X.size, n_outliers, replace=False), X.shape)
        X[indices] *= 10
        return X
