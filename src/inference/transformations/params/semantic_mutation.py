from .base import ParameterTransformation

class SemanticMutation(ParameterTransformation):
    """
    Performs semantic-level replacements for specific string hyperparameters.
    For example, switches 'gini' to 'entropy', or 'rbf' to 'linear'.
    """

    SEMANTIC_MAP = {
        "kernel": {"rbf": "linear", "linear": "poly", "poly": "sigmoid"},
        "criterion": {"gini": "entropy", "entropy": "gini"},
        "solver": {"lbfgs": "liblinear", "liblinear": "saga"},
    }

    def __init__(self, key: str):
        self.key = key

    def apply(self, params: dict) -> str | None:
        value = params.get(self.key)
        if value in self.SEMANTIC_MAP.get(self.key, {}):
            return self.SEMANTIC_MAP[self.key][value]
        return None

    @staticmethod
    def supports(value) -> bool:
        return isinstance(value, str)
