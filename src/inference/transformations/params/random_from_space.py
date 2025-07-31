"""Parameter transformation that randomly selects a value from the parameter search space."""

import random
from .base import ParameterTransformation

class RandomFromSpace(ParameterTransformation):
    """Randomly selects a value from a predefined set for the given parameter.

    This transformation perturbs the parameter by randomly choosing a new value
    from its search space, excluding the current value.

    Attributes:
        key (str): The parameter name to perturb.
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
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],

        # General
        "solver": ["lbfgs", "liblinear", "sag", "saga"],
        "multi_class": ["auto", "ovr", "multinomial"],
    }

    def __init__(self, key: str):
        """Initializes the RandomFromSpace transformation.

        Args:
            key (str): The parameter name to perturb.
        """
        self.key = key

    def apply(self, params: dict) -> str | None:
        """Applies random selection from search space to the parameter.

        Args:
            params (dict): Dictionary of model hyperparameters.

        Returns:
            str | None: New randomly selected value (if applied), else None.
        """
        space = self.PARAM_SPACE.get(self.key)
        current = params.get(self.key)
        if space and current in space:
            candidates = [val for val in space if val != current]
            if candidates:
                return random.choice(candidates)
        return None

    @staticmethod
    def supports(value) -> bool:
        """Checks if the value is a string.

        Args:
            value (Any): The parameter value to check.

        Returns:
            bool: True if value is a string, False otherwise.
        """
        return isinstance(value, str)
