import random
from .base import ParameterTransformation

class ScaleHyperparameter(ParameterTransformation):
    """
    Applies multiplicative perturbation to numeric hyperparameters.

    Useful for exploring robustness against changes in regularization strength,
    learning rates, or other scale-sensitive parameters.

    Example:
        C = 1.0  →  C = 0.5 or 1.5 (if factors = (0.5, 1.5))
    """

    def __init__(self, key: str, factors=(0.5, 1.5)):
        self.key = key
        self.factors = factors

    def apply(self, params: dict) -> float | int | None:
        value = params.get(self.key)
        if isinstance(value, (int, float)):
            factor = random.choice(self.factors)
            return round(value * factor, 10)
        return None

    @staticmethod
    def supports(value) -> bool:
        return isinstance(value, (int, float))
