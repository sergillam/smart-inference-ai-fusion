import random
from .base import ParameterTransformation

class BoundedNumericShift(ParameterTransformation):
    """
    Applies small additive perturbations to float hyperparameters, within a defined bound.

    Useful for parameters like:
    - alpha, gamma, learning_rate, var_smoothing, etc.
    """

    def __init__(self, key: str, shift_range: tuple = (-0.1, 0.1)):
        self.key = key
        self.shift_range = shift_range

    def apply(self, params: dict) -> float | None:
        value = params.get(self.key)
        if isinstance(value, float):
            shift = random.uniform(*self.shift_range)
            return max(0.0, value + shift)  # Avoid negative values
        return None

    @staticmethod
    def supports(value) -> bool:
        return isinstance(value, float)
