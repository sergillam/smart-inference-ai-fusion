from .base import ParameterTransformation

class BooleanFlip(ParameterTransformation):
    def __init__(self, key: str):
        self.key = key

    def apply(self, params: dict) -> dict:
        if self.key in params and isinstance(params[self.key], bool):
            params[self.key] = not params[self.key]
        return params

    @staticmethod
    def supports(value) -> bool:
        return isinstance(value, bool)
