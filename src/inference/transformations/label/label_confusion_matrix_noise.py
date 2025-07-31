"""Label transformation that applies confusion-matrix-based noise to labels."""

import numpy as np
from inference.transformations.label.base import LabelTransformation

class LabelConfusionMatrixNoise(LabelTransformation):
    """Applies confusion-matrix-based noise to the label vector.

    This transformation perturbs a fraction of labels, swapping classes according to
    a synthetic confusion matrix where each off-diagonal entry encodes the probability
    of mistaking one class for another.

    Attributes:
        noise_level (float): Fraction of labels to perturb.
    """

    def __init__(self, noise_level: float):
        """Initializes the confusion matrix noise transformation.

        Args:
            noise_level (float): Fraction of labels to be perturbed.
        """
        self.noise_level = noise_level

    def apply(self, y):
        """Applies confusion-matrix-based label noise to the input labels.

        Args:
            y (array-like): Original label vector.

        Returns:
            np.ndarray: Noisy label vector.
        """
        y = np.asarray(y)
        y_noisy = y.copy()
        n_samples = len(y)
        n_flip = int(n_samples * self.noise_level)
        if n_flip == 0:
            return y

        classes = np.unique(y)
        n_classes = len(classes)

        # Create artificial confusion matrix (high probability off-diagonal)
        confusion_matrix = np.full((n_classes, n_classes), 1.0 / (n_classes - 1))
        np.fill_diagonal(confusion_matrix, 0)

        # Normalize each row to sum to 1
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix /= row_sums

        indices = np.random.choice(n_samples, size=n_flip, replace=False)

        for idx in indices:
            current_class = y_noisy[idx]
            current_idx = np.where(classes == current_class)[0][0]
            # Sample a new class using the confusion probabilities
            new_class = np.random.choice(classes, p=confusion_matrix[current_idx])
            y_noisy[idx] = new_class

        return y_noisy
