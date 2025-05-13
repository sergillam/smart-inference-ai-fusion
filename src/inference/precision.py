import numpy as np
from inference.base_inference import InferenceTransformation

class TruncateDecimals(InferenceTransformation):
    def __init__(self, places):
        self.places = places

    def apply(self, X):
        return np.round(X, self.places)

class CastToInt(InferenceTransformation):
    def apply(self, X):
        return X.astype(int)

class Quantize(InferenceTransformation):
    def __init__(self, bins):
        self.bins = bins
    def apply(self, X):
        min_val, max_val = np.min(X), np.max(X)
        bins = np.linspace(min_val, max_val, self.bins + 1)
        return np.digitize(X, bins) - 1