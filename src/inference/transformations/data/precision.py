"""Transformations for numerical precision and discretization on input features."""

import numpy as np
from inference.transformations.data.base import InferenceTransformation

class TruncateDecimals(InferenceTransformation):
    """Rounds input data to a specified number of decimal places.

    Attributes:
        places (int): Number of decimal places to keep.
    """

    def __init__(self, places):
        """Initializes the transformation.

        Args:
            places (int): Number of decimal places for rounding.
        """
        self.places = places

    def apply(self, X):
        """Applies rounding to the input array.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Rounded feature matrix.
        """
        return np.round(X, self.places)

class CastToInt(InferenceTransformation):
    """Casts input features to integer type."""

    def apply(self, X):
        """Casts all elements in the input to integers.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with integer values.
        """
        return X.astype(int)

class Quantize(InferenceTransformation):
    """Discretizes input data into a fixed number of bins.

    Attributes:
        bins (int): Number of quantization bins.
    """

    def __init__(self, bins):
        """Initializes the transformation.

        Args:
            bins (int): Number of quantization bins.
        """
        self.bins = bins

    def apply(self, X):
        """Discretizes (quantizes) the input array into bins.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with quantized (binned) values.
        """
        min_val, max_val = np.min(X), np.max(X)
        bins = np.linspace(min_val, max_val, self.bins + 1)
        return np.digitize(X, bins) - 1
