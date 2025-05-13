import numpy as np
from inference.base_inference import InferenceTransformation

class AddDummyFeatures(InferenceTransformation):
    def __init__(self, num_features):
        self.num_features = num_features

    def apply(self, X):
        dummy = np.random.rand(X.shape[0], self.num_features)
        return np.concatenate([X, dummy], axis=1)
    
class DuplicateFeatures(InferenceTransformation):
    def __init__(self, num_duplicates):
        self.num_duplicates = num_duplicates
    def apply(self, X):
        duplicated = X[:, :self.num_duplicates]
        return np.concatenate([X, duplicated], axis=1)
