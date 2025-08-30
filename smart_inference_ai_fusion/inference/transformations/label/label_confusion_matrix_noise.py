"""Label transformation that applies confusion-matrix-based noise to labels."""

import numpy as np
from numpy.typing import ArrayLike

from .base import LabelTransformation


class LabelConfusionMatrixNoise(LabelTransformation):
    """Applies confusion-matrix-based noise to the label vector.

    This transformation perturbs a fraction of labels, swapping classes according to
    a synthetic confusion matrix where each off-diagonal entry encodes the probability
    of mistaking one class for another.

    Attributes:
        noise_level (float): Fraction of labels to perturb (between 0 and 1).
    """

    def __init__(self, noise_level: float):
        """Initialize the confusion matrix noise transformation.

        Args:
            noise_level (float): Fraction of labels to be perturbed (between 0 and 1).

        Raises:
            ValueError: If ``noise_level`` is not in the range [0, 1].
        """
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be between 0 and 1.")
        self.noise_level = noise_level

    def apply(self, y: ArrayLike) -> np.ndarray:
        """Apply confusion-matrix-based label noise to the input labels.

        Args:
            y (ArrayLike): Original label vector (1-D).
                Shape: (n_samples,).

        Returns:
            np.ndarray: Noisy label vector with the same shape as input.
                Shape: (n_samples,).
        """
        y = np.asarray(y)
        y_noisy = y.copy()
        n_samples = len(y)
        n_flip = int(n_samples * self.noise_level)
        if n_flip == 0:
            return y

        classes = np.unique(y)
        n_classes = len(classes)
        if n_classes < 2:
            # Nada a “confundir” se só existir uma classe
            return y

        # Artificial confusion matrix (off-diagonal probabilities uniformes)
        confusion_matrix = np.full((n_classes, n_classes), 1.0 / (n_classes - 1))
        np.fill_diagonal(confusion_matrix, 0.0)

        # Normalize rows (robustez contra possíveis efeitos numéricos)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = confusion_matrix / np.where(row_sums == 0, 1.0, row_sums)

        indices = np.random.choice(n_samples, size=n_flip, replace=False)

        for idx in indices:
            current_class = y_noisy[idx]
            current_idx = int(np.where(classes == current_class)[0][0])
            # Sample a new class using the confusion probabilities
            new_class = np.random.choice(classes, p=confusion_matrix[current_idx])
            y_noisy[idx] = new_class

        return y_noisy
