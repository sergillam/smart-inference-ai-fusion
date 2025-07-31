"""Base classes and abstractions for label inference transformations.

This module defines the abstract interface for all label (y) perturbation
and transformation techniques used in the inference framework.
"""

class LabelTransformation:
    """Abstract base class for label inference/perturbation transformations.

    Any subclass should implement the `apply` method to perform a transformation
    on the target labels.
    """

    def apply(self, y):
        """Applies the transformation to the input label vector.

        Args:
            y (array-like): Input label vector to be transformed.

        Returns:
            array-like: Transformed label vector.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement apply().")
