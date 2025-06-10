from .base import ParameterTransformation

class SemanticMutation(ParameterTransformation):
    """
    Performs semantic-level replacements for specific string hyperparameters.
    For example, switches 'gini' to 'entropy', or 'rbf' to 'linear'.
    """

    SEMANTIC_MAP = {
        # --- GaussianNB (poucos hiperparâmetros) ---
        "var_smoothing": {"1e-9": "1e-8", "1e-8": "1e-7", "1e-7": "1e-6", "1e-6": "1e-9"},
        
        # --- KNN ---
        "weights": {"uniform": "distance", "distance": "uniform"},
        "algorithm": {"auto": "ball_tree", "ball_tree": "kd_tree", "kd_tree": "brute", "brute": "auto"},
        "leaf_size": {"30": "50", "50": "10", "10": "30"},  # apenas como exemplo, normalmente é int

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
        "C": {"1.0": "10.0", "10.0": "0.1", "0.1": "1.0"},  # também é float normalmente

        # --- Perceptron ---
        "penalty": {"l2": "l1", "l1": "elasticnet", "elasticnet": "none", "none": "l2"},
        "fit_intercept": {"True": "False", "False": "True"},
        "shuffle": {"True": "False", "False": "True"},
        "early_stopping": {"True": "False", "False": "True"},
        "class_weight": {"balanced": "none", "none": "balanced"},
        "solver": {"lbfgs": "sgd", "sgd": "adam", "adam": "lbfgs"},  # para modelos que suportam esses solvers
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
