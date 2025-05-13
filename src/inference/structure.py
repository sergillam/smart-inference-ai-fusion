import numpy as np
from inference.base_inference import InferenceTransformation

class ShuffleFeatures(InferenceTransformation):
    def __init__(self, frac):
        self.frac = frac

    def apply(self, X):
        X_shuffled = X.copy()
        n_samples, n_features = X.shape
        n_shuffle = int(n_samples * self.frac)
        for i in range(n_features):
            idx = np.random.choice(n_samples, n_shuffle, replace=False)
            np.random.shuffle(X_shuffled[idx, i])
        return X_shuffled

class ScaleFeatures(InferenceTransformation):
    def __init__(self, scale_range):
        self.scale_range = scale_range

    def apply(self, X):
        return X * np.random.uniform(self.scale_range[0], self.scale_range[1], X.shape[1])

class RemoveFeatures(InferenceTransformation):
    def __init__(self, features_to_remove):
        self.features_to_remove = features_to_remove

    def apply(self, X):
        mask = np.ones(X.shape[1], dtype=bool)
        mask[self.features_to_remove] = False
        return X[:, mask]

class FeatureSwap(InferenceTransformation):
    def __init__(self, features_to_swap):
        self.features_to_swap = features_to_swap

    def apply(self, X):
        X_new = X.copy()
        for feature_idx in self.features_to_swap:
            shuffled = X_new[:, feature_idx].copy()
            np.random.shuffle(shuffled)
            X_new[:, feature_idx] = shuffled
        return X_new
