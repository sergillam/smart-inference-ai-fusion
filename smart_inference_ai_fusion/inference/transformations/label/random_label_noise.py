"""Transformation for injecting random label noise."""

import numpy as np
from numpy.typing import ArrayLike

from .base import LabelTransformation


class RandomLabelNoise(LabelTransformation):
    """Apply random label noise by flipping a fraction of labels to a different class.

    Attributes:
        flip_fraction (float): Fraction of labels to randomly flip (between 0 and 1).
    """

    def __init__(self, flip_fraction: float):
        """Initialize the RandomLabelNoise transformation.

        Args:
            flip_fraction (float): Fraction of labels to flip (between 0 and 1).

        Raises:
            ValueError: If ``flip_fraction`` is not in the range [0, 1].
        """
        if not 0.0 <= flip_fraction <= 1.0:
            raise ValueError("flip_fraction must be between 0 and 1.")
        self.flip_fraction = flip_fraction

    def apply(self, y: ArrayLike) -> np.ndarray:
        """Apply random label flipping to the input labels.

        Args:
            y (ArrayLike): Input label vector.
                Shape: (n_samples,).

        Returns:
            np.ndarray: Label vector with a fraction of randomly flipped labels.
                Shape: (n_samples,).

        Raises:
            ValueError: If there are fewer than two unique classes in ``y``.
        """
        y_np = np.asarray(y)
        if len(np.unique(y_np)) < 2:
            raise ValueError("Cannot flip labels when there is only one unique class.")

        y_noisy = y_np.copy()
        n_flips = int(len(y_np) * self.flip_fraction)
        indices = np.random.choice(len(y_np), n_flips, replace=False)
        classes = np.unique(y_np)

        for i in indices:
            available = classes[classes != y_noisy[i]]
            y_noisy[i] = np.random.choice(available)

        return y_noisy
