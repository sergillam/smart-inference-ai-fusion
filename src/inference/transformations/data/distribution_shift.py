"""Distribution shift transformation: simulates concept drift by mixing shifted samples."""

import numpy as np
from inference.transformations.data.base import InferenceTransformation

class DistributionShiftMixing(InferenceTransformation):
    """Simulates concept drift by replacing a fraction of samples with points
    drawn from a shifted distribution (e.g., mean shifted).

    Attributes:
        shift_fraction (float): Fraction of samples to replace with shifted samples.
        shift_strength (float): Multiplicative factor for mean shift.
    """

    def __init__(self, shift_fraction=0.1, shift_strength=2.0):
        """Initializes the distribution shift transformation.

        Args:
            shift_fraction (float, optional): Fraction of samples to shift. Default is 0.1.
            shift_strength (float, optional): Strength of the shift. Default is 2.0.
        """
        self.shift_fraction = shift_fraction
        self.shift_strength = shift_strength

    def apply(self, X):
        """Applies the distribution shift by replacing random samples with shifted points.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with shifted samples replacing some original samples.
        """
        n_samples = X.shape[0]
        n_shift = int(n_samples * self.shift_fraction)
        if n_shift == 0:
            return X

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        # Generate shifted samples from a new distribution
        shifted_samples = np.random.normal(
            loc=mean + self.shift_strength * std,
            scale=std,
            size=(n_shift, X.shape[1])
        )

        # Randomly replace samples in the original data
        x_new = X.copy()
        indices = np.random.choice(n_samples, n_shift, replace=False)
        x_new[indices] = shifted_samples
        return x_new
