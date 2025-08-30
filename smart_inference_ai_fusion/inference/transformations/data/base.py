"""Base classes and abstractions for data inference transformations.

This module defines abstract interfaces and base implementations for all
feature (X) perturbation and transformation techniques used in the inference framework.
"""

from abc import ABC, abstractmethod
from typing import Any


class InferenceTransformation(ABC):
    """Abstract base class for data inference/perturbation transformations.

    All transformations must inherit from this class and implement the
    `apply` method, which takes an input feature matrix and returns a
    transformed version.

    Example:
        >>> class GaussianNoiseTransform(InferenceTransformation):
        ...     def apply(self, X):
        ...         return X + np.random.normal(0, 0.1, X.shape)
    """

    @abstractmethod
    def apply(self, X: Any) -> Any:
        """Apply the transformation to the input data.

        Args:
            X (Any): Input feature matrix or array-like structure to transform.

        Returns:
            Any: Transformed feature matrix.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement apply().")
