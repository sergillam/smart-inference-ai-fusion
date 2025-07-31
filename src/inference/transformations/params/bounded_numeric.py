"""Parameter transformation for bounded numeric shift."""

import random
from .base import ParameterTransformation

class BoundedNumericShift(ParameterTransformation):
    """Applies an additive perturbation to a float hyperparameter within a defined bound.

    This transformation is useful for continuous hyperparameters such as
    'alpha', 'gamma', 'learning_rate', 'var_smoothing', etc.

    Attributes:
        key (str): Name of the parameter to be perturbed.
        shift_range (tuple): Minimum and maximum shift values.
    """

    def __init__(self, key: str, shift_range: tuple = (-0.1, 0.1)):
        """Initializes the BoundedNumericShift transformation.

        Args:
            key (str): Name of the parameter to be perturbed.
            shift_range (tuple, optional): Minimum and maximum shift values. 
                Defaults to (-0.1, 0.1).
        """
        self.key = key
        self.shift_range = shift_range

    def apply(self, params: dict) -> dict:
        """Applies the shift to the specified float parameter in-place.

        Args:
            params (dict): Dictionary of model hyperparameters.

        Returns:
            dict: Updated parameter dictionary with the shifted value.
        """
        params = params.copy()
        value = params.get(self.key)
        if isinstance(value, float):
            shift = random.uniform(*self.shift_range)
            params[self.key] = max(0.0, value + shift)  # Avoid negative values
        return params

    @staticmethod
    def supports(value) -> bool:
        """Checks if the value is a float.

        Args:
            value (Any): Value to check.

        Returns:
            bool: True if value is a float, False otherwise.
        """
        return isinstance(value, float)
