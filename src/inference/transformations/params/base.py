"""Base classes and abstractions for parameter inference transformations.

Defines the abstract interface for all parameter perturbation techniques used 
    in the inference framework.
"""

from abc import ABC, abstractmethod

class ParameterTransformation(ABC):
    """Abstract base class for parameter inference/perturbation transformations.

    Any subclass should implement the `apply` method to perform a transformation
    on the model parameters.

    Methods:
        apply(params): Applies the transformation to the parameter dictionary.
    """

    @abstractmethod
    def apply(self, params: dict) -> dict:
        """Applies the transformation to the model parameters.

        Args:
            params (dict): Dictionary of original model parameters.

        Returns:
            dict: Dictionary of perturbed model parameters.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement apply().")
