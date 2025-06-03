import random
from .base import ParameterTransformation

class TypeCastPerturbation(ParameterTransformation):
    """
    Randomly casts numeric or boolean parameters to another type to test model robustness.
    For example:
        - int → str
        - float → int
        - bool → str
    """

    def __init__(self, key: str):
        self.key = key

    def apply(self, params: dict) -> str | int | float | None:
        value = params.get(self.key)

        if isinstance(value, int):
            return str(value)  # int → str
        elif isinstance(value, float):
            return int(value)  # float → int
        elif isinstance(value, bool):
            return str(value)  # bool → str
        return None

    @staticmethod
    def supports(value) -> bool:
        return isinstance(value, (int, float, bool))
