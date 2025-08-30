"""Parameter transformation that performs semantic-level replacements for string hyperparameters."""

from .base import ParameterTransformation


class SemanticMutation(ParameterTransformation):
    """Performs semantic-level replacements for specific string hyperparameters.

    This transformation swaps a parameter's value for a semantically similar alternative,
    e.g., switching 'gini' to 'entropy', or 'rbf' to 'linear'.

    Attributes:
        key (str): The hyperparameter name to mutate.
    """

    SEMANTIC_MAP = {
        # --- GaussianNB (few hyperparameters) ---
        "var_smoothing": {"1e-9": "1e-8", "1e-8": "1e-7", "1e-7": "1e-6", "1e-6": "1e-9"},
        # --- KNN ---
        "weights": {"uniform": "distance", "distance": "uniform"},
        "algorithm": {
            "auto": "ball_tree",
            "ball_tree": "kd_tree",
            "kd_tree": "brute",
            "brute": "auto",
        },
        "leaf_size": {"30": "50", "50": "10", "10": "30"},  # Example; usually int
        # --- DecisionTreeClassifier ---
        "criterion": {"gini": "entropy", "entropy": "log_loss", "log_loss": "gini"},
        "splitter": {"best": "random", "random": "best"},
        "max_features": {"auto": "sqrt", "sqrt": "log2", "log2": "auto"},
        "class_weight": {"balanced": "none", "none": "balanced"},
        # --- SVM ---
        "kernel": {"rbf": "linear", "linear": "poly", "poly": "sigmoid", "sigmoid": "rbf"},
        "gamma": {"scale": "auto", "auto": "scale"},
        "shrinking": {"True": "False", "False": "True"},
        "decision_function_shape": {"ovr": "ovo", "ovo": "ovr"},
        "probability": {"True": "False", "False": "True"},
        "C": {"1.0": "10.0", "10.0": "0.1", "0.1": "1.0"},  # Example; usually float
        # --- Perceptron ---
        "penalty": {"l2": "l1", "l1": "elasticnet", "elasticnet": "none", "none": "l2"},
        "fit_intercept": {"True": "False", "False": "True"},
        "shuffle": {"True": "False", "False": "True"},
        "early_stopping": {"True": "False", "False": "True"},
        # "class_weight": {"balanced": "none", "none": "balanced"},
        "solver": {"lbfgs": "sgd", "sgd": "adam", "adam": "lbfgs"},
    }

    def __init__(self, key: str):
        """Initializes the semantic mutation transformation.

        Args:
            key (str): The hyperparameter name to mutate.
        """
        self.key = key

    def apply(self, params: dict) -> str | None:
        """Applies the semantic mutation to the parameter if defined in SEMANTIC_MAP.

        Args:
            params (dict): Dictionary of model hyperparameters.

        Returns:
            str | None: The mutated value if applicable, otherwise None.
        """
        value = params.get(self.key)
        if value in self.SEMANTIC_MAP.get(self.key, {}):
            return self.SEMANTIC_MAP[self.key][value]
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
