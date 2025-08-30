"""Outlier injection transformation for input data."""

import numpy as np

from .base import InferenceTransformation


class InjectOutliers(InferenceTransformation):
    """Injects outlier values into random elements of the input data.

    Attributes:
        frac (float): Fraction of data elements to convert into outliers.
    """

    def __init__(self, frac):
        """Initializes the InjectOutliers transformation.

        Args:
            frac (float): Fraction of data elements to multiply and turn into outliers.
        """
        self.frac = frac

    def apply(self, X):
        """Applies outlier injection to a random subset of the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with injected outliers.
        """
        n_samples, n_features = X.shape
        n_outliers = int(n_samples * n_features * self.frac)
        indices = np.unravel_index(np.random.choice(X.size, n_outliers, replace=False), X.shape)
        X = X.copy()
        X[indices] *= 10
        return X
