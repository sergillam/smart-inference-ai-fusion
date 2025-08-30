"""Data corruption transformations: zeroing out and inserting NaN values."""

import numpy as np

from .base import InferenceTransformation


class ZeroOut(InferenceTransformation):
    """Randomly sets a fraction of values in the input array to zero.

    Attributes:
        frac (float): Fraction of elements to set to zero.
    """

    def __init__(self, frac):
        """Initializes the ZeroOut transformation.

        Args:
            frac (float): Fraction of values to set to zero (between 0 and 1).
        """
        self.frac = frac

    def apply(self, X):
        """Applies zero-out corruption to the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with a fraction of values set to zero.
        """
        mask = np.random.rand(*X.shape) < self.frac
        X = X.copy()
        X[mask] = 0.0
        return X


class InsertNaN(InferenceTransformation):
    """Randomly inserts NaN values into the input array.

    Attributes:
        frac (float): Fraction of elements to set to NaN.
    """

    def __init__(self, frac):
        """Initializes the InsertNaN transformation.

        Args:
            frac (float): Fraction of values to set as NaN (between 0 and 1).
        """
        self.frac = frac

    def apply(self, X):
        """Applies NaN corruption to the input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with randomly inserted NaN values.
        """
        X = X.copy()
        n_samples, n_features = X.shape
        n_missing = int(n_samples * n_features * self.frac)
        if n_missing == 0:
            return X
        indices = np.unravel_index(np.random.choice(X.size, n_missing, replace=False), X.shape)
        X[indices] = np.nan
        return X
