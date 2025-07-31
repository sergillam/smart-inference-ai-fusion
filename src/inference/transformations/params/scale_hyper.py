"""Parameter transformation that applies multiplicative scaling to numeric hyperparameters."""

import random
from .base import ParameterTransformation

class ScaleHyperparameter(ParameterTransformation):
    """Applies multiplicative perturbation to numeric hyperparameters.

    This transformation multiplies the parameter value by a random scaling
    factor to test robustness against changes in scale-sensitive parameters.

    Attributes:
        key (str): The parameter name to perturb.
        factors (tuple): Tuple of possible scaling factors.
    """

    def __init__(self, key: str, factors=(0.5, 1.5)):
        """Initializes the ScaleHyperparameter transformation.

        Args:
            key (str): The parameter name to perturb.
            factors (tuple, optional): Scaling factors to choose from (default is (0.5, 1.5)).
        """
        self.key = key
        self.factors = factors

    def apply(self, params: dict) -> float | int | None:
        """Applies multiplicative scaling to the parameter value.

        Args:
            params (dict): Dictionary of model hyperparameters.

        Returns:
            float | int | None: The perturbed value, or None if not applicable.
        """
        value = params.get(self.key)
        if isinstance(value, (int, float)):
            factor = random.choice(self.factors)
            return round(value * factor, 10)
        return None

    @staticmethod
    def supports(value) -> bool:
        """Checks if the value is a numeric type (int or float).

        Args:
            value (Any): The parameter value to check.

        Returns:
            bool: True if value is int or float, False otherwise.
        """
        return isinstance(value, (int, float))
