"""Base classes and abstractions for label inference transformations.

This module defines the abstract interface for all label (y) perturbation
and transformation techniques used in the inference framework.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class LabelTransformation(ABC):
    """Abstract base class for label inference/perturbation transformations.

    Any subclass must implement :meth:`apply` to perform a transformation on target labels.
    """

    @abstractmethod
    def apply(self, y: ArrayLike) -> np.ndarray:
        """Apply the transformation to the input label vector.

        Args:
            y (ArrayLike): Input label vector to be transformed.
                Shape: (n_samples,).

        Returns:
            np.ndarray: Transformed label vector with the same shape as the input.
                Shape: (n_samples,).
        """
        raise NotImplementedError("Subclasses must implement apply().")
