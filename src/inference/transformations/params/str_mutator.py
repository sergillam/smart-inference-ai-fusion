from .base import ParameterTransformation

class StringMutator(ParameterTransformation):
    def __init__(self, key: str, options=None):
        self.key = key
        self.options = options or ["gini", "entropy", "log_loss"]

    def apply(self, params: dict) -> dict:
        if self.key in params and isinstance(params[self.key], str):
            current = params[self.key]
            choices = [opt for opt in self.options if opt != current]
            if choices:
                import random
                params[self.key] = random.choice(choices)
        return params

    @staticmethod
    def supports(value) -> bool:
        return isinstance(value, str)
