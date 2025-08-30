"""Label transformation that simulates partial ambiguity in class labels."""

import numpy as np
from numpy.typing import ArrayLike

from .base import LabelTransformation


class PartialLabelNoise(LabelTransformation):
    """Simulates ambiguity in labels by replacing a fraction with an alternative valid label.

    Instead of returning a multi-label string (e.g., '3|0'), this transformation
    randomly chooses between the true and an alternative label, preserving the original type.

    Attributes:
        noise_fraction (float): Fraction of labels to perturb (between 0 and 1).
    """

    def __init__(self, noise_fraction: float):
        """Initialize the PartialLabelNoise transformation.

        Args:
            noise_fraction (float): Fraction of labels to replace (between 0 and 1).
        """
        self.noise_fraction = noise_fraction

    def apply(self, y: ArrayLike) -> np.ndarray:
        """Apply partial label noise by randomly replacing labels with alternatives.

        Args:
            y (ArrayLike): Input label vector.
                Shape: (n_samples,).

        Returns:
            np.ndarray: Label vector with a fraction of ambiguous/altered labels.
                Shape: (n_samples,).

        Raises:
            ValueError: If ``noise_fraction`` is outside the range [0, 1].
        """
        if not 0 <= self.noise_fraction <= 1:
            raise ValueError("noise_fraction must be between 0 and 1")

        y_np = np.asarray(y)
        y_noisy = y_np.copy()

        n = int(len(y_np) * self.noise_fraction)
        indices = np.random.choice(len(y_np), n, replace=False)
        classes = np.unique(y_np)

        for idx in indices:
            true_label = y_noisy[idx]
            alternatives = classes[classes != true_label]
            alt_label = np.random.choice(alternatives)
            y_noisy[idx] = np.random.choice([true_label, alt_label])

        return y_noisy
