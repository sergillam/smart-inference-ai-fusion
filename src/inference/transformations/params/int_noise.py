import random
from .base import ParameterTransformation

class IntegerNoise(ParameterTransformation):
    def __init__(self, key: str, delta: int = 2):
        self.key = key
        self.delta = delta

    def apply(self, params: dict) -> dict:
        if self.key in params and isinstance(params[self.key], int):
            params[self.key] += random.randint(-self.delta, self.delta)
        return params

    @staticmethod
    def supports(value) -> bool:
        return isinstance(value, int)
