"""Base classes and abstractions for data inference transformations.

This module defines abstract interfaces and base implementations for all
feature (X) perturbation and transformation techniques used in the inference framework.
"""

class InferenceTransformation:
    """Abstract base class for data inference/perturbation transformations.

    Any subclass should implement the `apply` method to perform a transformation
    on the input data.
    """

    def apply(self, X):
        """Applies the transformation to the input data.

        Args:
            X (array-like): Input features to be transformed.

        Returns:
            array-like: Transformed features.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement apply().")
