"""Transformation for injecting random label noise."""

import numpy as np
from inference.transformations.label.base import LabelTransformation

class RandomLabelNoise(LabelTransformation):
    """Applies random label noise by flipping a fraction of labels to a different class.

    Attributes:
        flip_fraction (float): Fraction of labels to randomly flip.
    """

    def __init__(self, flip_fraction):
        """Initializes the RandomLabelNoise transformation.

        Args:
            flip_fraction (float): Fraction of labels to flip (between 0 and 1).
        """
        self.flip_fraction = flip_fraction

    def apply(self, y):
        """Applies random label flipping to the input labels.

        Args:
            y (array-like): Input label vector.

        Returns:
            np.ndarray: Label vector with a fraction of randomly flipped labels.
        """
        y_np = np.asarray(y)
        y_noisy = y_np.copy()
        n = int(len(y_np) * self.flip_fraction)
        indices = np.random.choice(len(y_np), n, replace=False)
        classes = np.unique(y_np)
        for i in indices:
            available = classes[classes != y_noisy[i]]
            y_noisy[i] = np.random.choice(available)
        return y_noisy
