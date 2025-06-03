import random
from .base import ParameterTransformation

class RandomFromSpace(ParameterTransformation):
    """
    Randomly selects a value from a predefined set for the given parameter.

    Example:
        kernel ∈ {rbf, linear, poly, sigmoid}
    """

    PARAM_SPACE = {
        # SVM
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "gamma": ["scale", "auto"],
        "shrinking": [True, False],

        # DecisionTreeClassifier
        "criterion": ["gini", "entropy", "log_loss"],
        "splitter": ["best", "random"],
        "max_features": ["auto", "sqrt", "log2", None],
        "class_weight": ["balanced", None],

        # Perceptron
        "penalty": ["l2", "l1", "elasticnet", None],
        "fit_intercept": [True, False],
        "shuffle": [True, False],
        "early_stopping": [True, False],

        # KNN
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [10, 20, 30, 40, 50],

        # GaussianNB
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],  # valores típicos para teste

        # General
        "solver": ["lbfgs", "liblinear", "sag", "saga"],
        "multi_class": ["auto", "ovr", "multinomial"],
    }

    def __init__(self, key: str):
        self.key = key

    def apply(self, params: dict) -> str | None:
        space = self.PARAM_SPACE.get(self.key)
        current = params.get(self.key)
        if space and current in space:
            candidates = [val for val in space if val != current]
            if candidates:
                return random.choice(candidates)
        return None

    @staticmethod
    def supports(value) -> bool:
        return isinstance(value, str)
