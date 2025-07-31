"""Transformation for injecting temporal drift into sequential data."""

import numpy as np
from inference.transformations.data.base import InferenceTransformation

class TemporalDriftInjection(InferenceTransformation):
    """Simulates temporal drift by applying progressively increasing noise to samples.

    This is useful for sequential or sensor data, where drift accumulates over time.

    Attributes:
        drift_std (float): Standard deviation multiplier for the drift noise.
    """

    def __init__(self, drift_std=0.05):
        """Initializes the TemporalDriftInjection transformation.

        Args:
            drift_std (float, optional): Standard deviation for drift noise (default is 0.05).
        """
        self.drift_std = drift_std

    def apply(self, X):
        """Applies temporal drift to the feature matrix.

        Noise is added cumulatively to each sample, increasing with sample index.

        Args:
            X (np.ndarray): Input feature matrix, assumed to be ordered temporally.

        Returns:
            np.ndarray: Feature matrix with temporal drift applied.
        """
        n_samples, n_features = X.shape
        x_drifted = X.copy()

        for i in range(n_samples):
            drift = np.random.normal(
                loc=0.0, scale=self.drift_std * (i / n_samples), size=n_features
            )
            x_drifted[i] += drift

        return x_drifted
