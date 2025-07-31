"""Feature distraction transformations: add or duplicate features in the input array."""

import numpy as np
from inference.transformations.data.base import InferenceTransformation

class AddDummyFeatures(InferenceTransformation):
    """Adds randomly generated dummy features to the input array.

    Attributes:
        num_features (int): Number of dummy features to add.
    """

    def __init__(self, num_features):
        """Initializes the AddDummyFeatures transformation.

        Args:
            num_features (int): Number of dummy features to add.
        """
        self.num_features = num_features

    def apply(self, X):
        """Appends dummy features with random values to the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with additional dummy features.
        """
        dummy = np.random.rand(X.shape[0], self.num_features)
        return np.concatenate([X, dummy], axis=1)

class DuplicateFeatures(InferenceTransformation):
    """Duplicates a specified number of features from the input array.

    Attributes:
        num_duplicates (int): Number of leading features to duplicate.
    """

    def __init__(self, num_duplicates):
        """Initializes the DuplicateFeatures transformation.

        Args:
            num_duplicates (int): Number of leading features to duplicate.
        """
        self.num_duplicates = num_duplicates

    def apply(self, X):
        """Appends duplicates of the first `num_duplicates` features to the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with duplicated features appended.
        """
        duplicated = X[:, :self.num_duplicates]
        return np.concatenate([X, duplicated], axis=1)
