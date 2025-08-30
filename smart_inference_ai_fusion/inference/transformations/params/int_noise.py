"""Parameter transformation for injecting random integer noise."""

import random

from .base import ParameterTransformation


class IntegerNoise(ParameterTransformation):
    """Applies small random integer noise to an integer hyperparameter.

    This transformation perturbs the given parameter by adding or subtracting
    a random integer within the specified delta range.

    Attributes:
        key (str): Name of the hyperparameter to perturb.
        delta (int): Maximum absolute value of noise to add or subtract.
    """

    def __init__(self, key: str, delta: int = 2):
        """Initializes the IntegerNoise transformation.

        Args:
            key (str): Name of the hyperparameter to perturb.
            delta (int, optional): Range for random perturbation (default is 2).
        """
        self.key = key
        self.delta = delta

    def apply(self, params: dict) -> dict:
        """Applies integer noise to the selected parameter if it is an integer.

        Args:
            params (dict): Dictionary of model hyperparameters.

        Returns:
            dict: Updated dictionary with the perturbed integer parameter.
        """
        params = params.copy()
        if self.key in params and isinstance(params[self.key], int):
            params[self.key] += random.randint(-self.delta, self.delta)
        return params

    @staticmethod
    def supports(value) -> bool:
        """Checks if the value is an integer.

        Args:
            value (Any): Value to check.

        Returns:
            bool: True if value is an integer.
        """
        return isinstance(value, int)
