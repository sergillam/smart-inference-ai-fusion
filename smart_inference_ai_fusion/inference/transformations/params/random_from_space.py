"""Parameter transformation that randomly selects a value from the parameter search space."""

import random

from smart_inference_ai_fusion.utils.report import ReportMode, report_data

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
        # General solvers (for Ridge, Logistic, etc.)
        "solver": ["lbfgs", "liblinear", "sag", "saga"],
        # MLP-specific solver (only valid options)
        "solver_mlp": ["adam", "lbfgs", "sgd"],
        # MLP-specific learning_rate (only valid options)
        "learning_rate": ["constant", "invscaling", "adaptive"],
        # Ridge-specific solver (only valid options)
        "solver_ridge": ["lsqr", "saga", "sparse_cg", "auto", "svd", "lbfgs", "cholesky", "sag"],
        "multi_class": ["auto", "ovr", "multinomial"],
        # 🧪 SCIENTIFIC PROTECTION: Valid n_jobs values (no 0)
        "n_jobs": [-1, 1, 2, 4, 8],  # -1 = use all cores, positive integers only
        # 🧪 SCIENTIFIC PROTECTION: Valid max_depth values (no 0)
        "max_depth": [None, 3, 5, 7, 10, 15, 20],  # None = unlimited, positive integers only
        # 🧪 SCIENTIFIC PROTECTION: Valid clustering parameters (no 0)
        "n_clusters": [2, 3, 4, 5, 8, 10, 15, 20],  # must be >= 1
        "n_components": [1, 2, 3, 4, 5, 8, 10, 15, 20],  # must be >= 1
        "n_estimators": [10, 25, 50, 100, 200],  # must be >= 1
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
        # Handle solver and learning_rate protection first
        if self.key == "solver":
            return self._handle_solver_protection(params)

        if self.key == "learning_rate":
            return self._handle_learning_rate_protection(params)

        # Standard parameter mutation
        return self._apply_standard_mutation(params)

    def _apply_standard_mutation(self, params: dict) -> str | None:
        """Apply standard parameter mutation logic.

        Args:
            params (dict): Dictionary of model hyperparameters.

        Returns:
            str | None: New parameter value or None.
        """
        if self.key not in self.PARAM_SPACE or self.key not in params:
            return None

        space = self.PARAM_SPACE[self.key]
        current_value = params[self.key]

        if current_value not in space:
            return None

        candidates = [val for val in space if val != current_value]
        if not candidates:
            return None

        new_value = random.choice(candidates)

        report_data(
            f"🧪 SCIENTIFIC PERTURBATION: Parameter mutation "
            f"{self.key}='{current_value}' -> '{new_value}' (testing robustness)",
            mode=ReportMode.PRINT,
        )
        return new_value

    def _handle_solver_protection(self, params: dict) -> str | None:
        """Handle model-specific solver validation.

        Args:
            params (dict): Dictionary of model hyperparameters.

        Returns:
            str | None: New solver value or None.
        """
        # MLPClassifier has hidden_layer_sizes (unique identifier)
        if "hidden_layer_sizes" in params:
            return self._apply_solver_mutation(params, "solver_mlp", "MLP")

        # RidgeClassifier detection (has alpha but NOT hidden_layer_sizes)
        if "alpha" in params and "hidden_layer_sizes" not in params:
            return self._apply_solver_mutation(params, "solver_ridge", "Ridge")

        # For other models, skip solver mutation to avoid confusion
        return None

    def _apply_solver_mutation(self, params: dict, space_key: str, model_type: str) -> str | None:
        """Apply solver mutation for a specific model type.

        Args:
            params (dict): Dictionary of model hyperparameters.
            space_key (str): Key for the solver space in PARAM_SPACE.
            model_type (str): Model type name for logging.

        Returns:
            str | None: New solver value or None.
        """
        space = self.PARAM_SPACE.get(space_key, [])
        current = params.get(self.key)

        if current not in space:
            return None

        candidates = [val for val in space if val != current]
        if not candidates:
            return None

        new_value = random.choice(candidates)
        report_data(
            f"🧪 SCIENTIFIC PERTURBATION: Applying {model_type} solver mutation "
            f"solver='{current}' -> '{new_value}' (testing robustness)",
            mode=ReportMode.PRINT,
        )
        return new_value

    def _handle_learning_rate_protection(self, params: dict) -> str | None:
        """Handle learning_rate validation for MLPClassifier.

        Args:
            params (dict): Dictionary of model hyperparameters.

        Returns:
            str | None: New learning_rate value or None.
        """
        # MLPClassifier has hidden_layer_sizes (unique identifier)
        if "hidden_layer_sizes" in params:
            return self._apply_learning_rate_mutation(params, "learning_rate", "MLP")

        # For other models, skip learning_rate mutation to avoid confusion
        return None

    def _apply_learning_rate_mutation(self, params: dict, key: str, model_type: str) -> str | None:
        """Apply learning_rate mutation for specific model type.

        Args:
            params (dict): Dictionary of model hyperparameters.
            key (str): Parameter key to mutate.
            model_type (str): Type of model for logging.

        Returns:
            str | None: New parameter value or None.
        """
        space = self.PARAM_SPACE.get(key, ["constant", "invscaling", "adaptive"])
        current = params.get(self.key)

        if current not in space:
            return None

        candidates = [val for val in space if val != current]
        if not candidates:
            return None

        new_value = random.choice(candidates)
        report_data(
            f"🧪 SCIENTIFIC PERTURBATION: Applying {model_type} learning_rate mutation "
            f"learning_rate='{current}' -> '{new_value}' (testing robustness)",
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
