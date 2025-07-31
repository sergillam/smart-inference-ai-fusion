"""Conditional noise transformation for input features.

Adds Gaussian noise only to samples meeting a specified feature-based condition.
"""

import numpy as np
from inference.transformations.data.base import InferenceTransformation

class ConditionalNoise(InferenceTransformation):
    """Applies Gaussian noise to samples that meet a condition on a specified feature.

    This transformation perturbs only those samples for which the value of a selected feature
    exceeds a given threshold.

    Attributes:
        feature_index (int): Index of the feature to apply the condition on.
        threshold (float): Threshold value for the feature condition.
        noise_std (float): Standard deviation of the Gaussian noise to apply.
    """

    def __init__(self, feature_index: int, threshold: float, noise_std: float = 0.1):
        """Initializes the ConditionalNoise transformation.

        Args:
            feature_index (int): Index of the feature to check the condition.
            threshold (float): Threshold value for the condition.
            noise_std (float, optional): Standard deviation of the noise. Defaults to 0.1.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.noise_std = noise_std

    def apply(self, X):
        """Applies conditional Gaussian noise to input features.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Perturbed feature matrix with conditional noise applied.
        """
        X = np.array(X, dtype=np.float32)
        condition = X[:, self.feature_index] > self.threshold
        noise = np.random.normal(0, self.noise_std, size=X.shape)
        X[condition] += noise[condition]
        return X
