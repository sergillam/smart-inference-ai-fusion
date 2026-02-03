"""Parameter transformation that performs semantic-level replacements for string hyperparameters."""

from smart_inference_ai_fusion.utils.report import ReportMode, report_data

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

    def apply(self, params: dict) -> str:
        """Apply semantic mutation to a string parameter.

        Args:
            params (dict): Dictionary containing all parameters.

        Returns:
            str or None: The mutated value or None if not applicable.
        """
        if self.key not in params:
            return None

        value = params[self.key]
        if not isinstance(value, str) or self.key not in self.SEMANTIC_MAP:
            return None

        # Handle solver protection for specific models
        if self.key == "solver":
            return self._handle_solver_mutation(params, value)

        # Standard semantic mutation for other parameters
        return self._apply_standard_mutation(value)

    def _handle_solver_mutation(self, params: dict, value: str) -> str | None:
        """Handle model-specific solver mutation.

        Args:
            params (dict): Dictionary containing all parameters.
            value (str): Current solver value.

        Returns:
            str | None: New solver value or None.
        """
        model_config = self._detect_model_solver_map(params)
        if model_config is None:
            return None

        solver_map, model_name = model_config
        return self._apply_solver_semantic_mutation(value, solver_map, model_name)

    def _detect_model_solver_map(self, params: dict) -> tuple | None:
        """Detect model type and return solver mapping.

        Returns:
            tuple: (solver_map, model_name) or None
        """
        # MLPClassifier has hidden_layer_sizes (unique identifier)
        if "hidden_layer_sizes" in params:
            return ({"adam": "lbfgs", "lbfgs": "sgd", "sgd": "adam"}, "MLP")
        # RidgeClassifier detection (has alpha but NOT hidden_layer_sizes and NOT C parameter)
        if "alpha" in params and "hidden_layer_sizes" not in params and "C" not in params:
            return ({"saga": "lbfgs", "lbfgs": "auto", "auto": "svd", "svd": "saga"}, "Ridge")
        # LogisticRegression detection (has C parameter but NOT alpha or hidden_layer_sizes)
        if "C" in params and "alpha" not in params and "hidden_layer_sizes" not in params:
            return (
                {"lbfgs": "liblinear", "liblinear": "sag", "sag": "saga", "saga": "lbfgs"},
                "LogisticRegression",
            )
        return None

    def _apply_solver_semantic_mutation(
        self, value: str, solver_map: dict, model_name: str
    ) -> str | None:
        """Apply semantic mutation to solver value."""
        if value not in solver_map:
            return None
        new_value = solver_map[value]
        report_data(
            f"🧪 SCIENTIFIC PERTURBATION: Applying {model_name} semantic mutation "
            f"solver='{value}' -> '{new_value}' (testing algorithmic robustness)",
            mode=ReportMode.PRINT,
        )
        return new_value

    def _apply_standard_mutation(self, value: str) -> str | None:
        """Apply standard semantic mutation.

        Args:
            value (str): Current parameter value.

        Returns:
            str | None: New parameter value or None.
        """
        if value not in self.SEMANTIC_MAP[self.key]:
            return None

        new_value = self.SEMANTIC_MAP[self.key][value]

        # Add scientific perturbation message for solver changes
        if self.key == "solver":
            report_data(
                f"🧪 SCIENTIFIC PERTURBATION: Applying semantic mutation "
                f"solver='{value}' -> '{new_value}' (testing algorithmic robustness)",
                mode=ReportMode.PRINT,
            )
        return new_value

    @staticmethod
    def supports(value) -> bool:
        """Checks if the value is a string.

        Args:
            value (Any): The parameter value to check.

        Returns:
            bool: True if value is a string, False otherwise.
        """
        return isinstance(value, str)
