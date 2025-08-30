"""Transformations for feature-level structure perturbations in tabular data."""

import numpy as np

from smart_inference_ai_fusion.inference.transformations.data.base import InferenceTransformation


class ShuffleFeatures(InferenceTransformation):
    """Randomly shuffles values within features for a fraction of the samples.

    Attributes:
        frac (float): Fraction of samples to shuffle within each feature.
    """

    def __init__(self, frac):
        """Initializes the ShuffleFeatures transformation.

        Args:
            frac (float): Fraction of samples to shuffle (between 0 and 1).
        """
        self.frac = frac

    def apply(self, X):
        """Applies the shuffling to a fraction of each feature's values.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with shuffled feature values.
        """
        x_shuffled = X.copy()
        n_samples, n_features = X.shape
        n_shuffle = int(n_samples * self.frac)
        for i in range(n_features):
            idx = np.random.choice(n_samples, n_shuffle, replace=False)
            np.random.shuffle(x_shuffled[idx, i])
        return x_shuffled


class ScaleFeatures(InferenceTransformation):
    """Scales each feature by a random factor within a specified range.

    Attributes:
        scale_range (tuple): (min, max) range for scaling factors.
    """

    def __init__(self, scale_range):
        """Initializes the ScaleFeatures transformation.

        Args:
            scale_range (tuple of float): (min, max) range for scaling each feature.
        """
        self.scale_range = scale_range

    def apply(self, X):
        """Applies random scaling to all features.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Scaled feature matrix.
        """
        return X * np.random.uniform(self.scale_range[0], self.scale_range[1], X.shape[1])


class RemoveFeatures(InferenceTransformation):
    """Removes specific features (columns) from the input data.

    Attributes:
        features_to_remove (list): List of feature indices to remove.
    """

    def __init__(self, features_to_remove):
        """Initializes the RemoveFeatures transformation.

        Args:
            features_to_remove (list of int): Indices of features to remove.
        """
        self.features_to_remove = features_to_remove

    def apply(self, X):
        """Removes the selected features from the data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with specified features removed.
        """
        mask = np.ones(X.shape[1], dtype=bool)
        mask[self.features_to_remove] = False
        return X[:, mask]


class FeatureSwap(InferenceTransformation):
    """Randomly permutes the values of selected features across samples.

    Attributes:
        features_to_swap (list): List of feature indices to swap.
    """

    def __init__(self, features_to_swap):
        """Initializes the FeatureSwap transformation.

        Args:
            features_to_swap (list of int): Indices of features to permute.
        """
        self.features_to_swap = features_to_swap

    def apply(self, X):
        """Randomly permutes values of selected features.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with selected features swapped.
        """
        x_new = X.copy()
        for feature_idx in self.features_to_swap:
            shuffled = x_new[:, feature_idx].copy()
            np.random.shuffle(shuffled)
            x_new[:, feature_idx] = shuffled
        return x_new
