"""Noise-based inference transformations for input data."""

import numpy as np

from smart_inference_ai_fusion.inference.transformations.data.base import InferenceTransformation


class GaussianNoise(InferenceTransformation):
    """Adds Gaussian noise to all features of the input data.

    Attributes:
        level (float): Standard deviation of the Gaussian noise.
    """

    def __init__(self, level):
        """Initializes the GaussianNoise transformation.

        Args:
            level (float): Standard deviation of the Gaussian noise to add.
        """
        self.level = level

    def apply(self, X):
        """Applies Gaussian noise to the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Noisy feature matrix.
        """
        return X + np.random.normal(0, self.level, X.shape)


class FeatureSelectiveNoise(InferenceTransformation):
    """Adds Gaussian noise to a specific subset of features.

    Attributes:
        level (float): Standard deviation of the Gaussian noise.
        features (list of int): Indices of features to perturb.
    """

    def __init__(self, level, features):
        """Initializes the FeatureSelectiveNoise transformation.

        Args:
            level (float): Standard deviation of the Gaussian noise to add.
            features (list of int): List of feature indices to apply noise to.
        """
        self.level = level
        self.features = features

    def apply(self, X):
        """Applies Gaussian noise to selected features of the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with selective noise.
        """
        x_new = X.copy()
        for feature_idx in self.features:
            x_new[:, feature_idx] += np.random.normal(0, self.level, X.shape[0])
        return x_new
