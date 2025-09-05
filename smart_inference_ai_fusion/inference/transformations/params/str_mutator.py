"""Parameter transformation that mutates a string hyperparameter among valid options."""

import logging
import random

from smart_inference_ai_fusion.inference.transformations.params.cross_dependency import (
    CrossDependencyPerturbation,
)
from smart_inference_ai_fusion.utils.report import (
    ReportMode,
    report_data,
)

from .base import ParameterTransformation

logger = logging.getLogger(__name__)


class StringMutator(ParameterTransformation):
    """Randomly mutates a string hyperparameter among provided options.

    Useful for categorical hyperparameters, with parameter-specific valid options.

    Attributes:
        key (str): The hyperparameter name to mutate.
        options (list): List of valid options to choose from.
    """

    # Valid options for different parameters
    PARAMETER_OPTIONS = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_features": ["sqrt", "log2", None],  # GradientBoosting-safe options
        "splitter": ["best", "random"],
        "kernel": ["rbf", "linear", "poly", "sigmoid"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "weights": ["uniform", "distance"],
        "class_weight": ["balanced", None],
        "solver": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
        "penalty": ["l1", "l2", "elasticnet", None],
        "learning_rate": [
            "constant",
            "invscaling",
            "adaptive",
        ],  # 🧪 FIXED: removed 'optimal' (invalid for MLPClassifier)
        "loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
        "multi_class": ["ovr", "multinomial"],
        "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine", "nearest_neighbors"],
        "linkage": ["ward", "complete", "average", "single"],
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "assign_labels": ["kmeans", "discretize", "cluster_qr"],
        # MLP-specific parameters
        "activation": ["relu", "tanh", "logistic"],
        "solver_mlp": ["adam", "lbfgs", "sgd"],
    }

    def __init__(self, key: str, options=None):
        """Initializes the StringMutator transformation.

        Args:
            key (str): The hyperparameter name to mutate.
            options (list, optional): List of possible string values.
                If None, uses parameter-specific options from PARAMETER_OPTIONS.
        """
        self.key = key
        if options is not None:
            self.options = options
        else:
            # Use parameter-specific options if available
            self.options = self.PARAMETER_OPTIONS.get(key, ["gini", "entropy", "log_loss"])

    def _handle_max_features_protection(self, params: dict, current) -> dict:
        """Handle GradientBoosting max_features protection."""
        gb_indicators = {"learning_rate", "subsample", "n_estimators"}
        if any(indicator in params for indicator in gb_indicators):
            safe_options = ["sqrt", "log2", None]
            if current not in safe_options:
                new_value = random.choice(safe_options)
                logger.warning(
                    "🧪 SCIENTIFIC PROTECTION: GradientBoosting detected, using safe "
                    "max_features '%s' -> '%s' (preventing InvalidParameterError)",
                    current,
                    new_value,
                )
                params[self.key] = new_value
                return params
        return None

    def _handle_solver_protection(self, params: dict, current) -> dict:
        """Handle MLPClassifier and RidgeClassifier solver protection."""
        # MLPClassifier has hidden_layer_sizes (unique identifier)
        if "hidden_layer_sizes" in params:
            safe_options = ["adam", "lbfgs", "sgd"]
            if current not in safe_options:
                new_value = random.choice(safe_options)
                report_data(
                    f"🧪 SCIENTIFIC PROTECTION: MLPClassifier detected, using safe "
                    f"solver '{current}' -> '{new_value}' (preventing InvalidParameterError)",
                    mode=ReportMode.PRINT,
                )
                params[self.key] = new_value
                return params
            # Valid MLP solver, allow mutation within safe options
            choices = [opt for opt in safe_options if opt != current]
            if choices:
                new_value = random.choice(choices)
                report_data(
                    f"🧪 SCIENTIFIC PERTURBATION: Applying MLP solver mutation "
                    f"solver='{current}' -> '{new_value}' (testing robustness)",
                    mode=ReportMode.PRINT,
                )
                params[self.key] = new_value
                validator = CrossDependencyPerturbation()
                params = validator.apply(params)

                return params
        # RidgeClassifier detection (has alpha but NOT hidden_layer_sizes)
        elif "alpha" in params and "hidden_layer_sizes" not in params:
            safe_options = ["saga", "lbfgs", "auto", "svd", "cholesky", "sag", "lsqr", "sparse_cg"]
            if current not in safe_options:
                new_value = random.choice(safe_options)
                report_data(
                    f"🧪 SCIENTIFIC PROTECTION: RidgeClassifier detected, using safe "
                    f"solver '{current}' -> '{new_value}' (preventing InvalidParameterError)",
                    mode=ReportMode.PRINT,
                )
                params[self.key] = new_value
                return params
            # Valid Ridge solver, allow mutation within safe options
            # But first check if cross-dependency validation should be applied
            choices = [opt for opt in safe_options if opt != current]
            if choices:
                new_value = random.choice(choices)
                report_data(
                    f"🧪 SCIENTIFIC PERTURBATION: Applying Ridge solver mutation "
                    f"solver='{current}' -> '{new_value}' (testing robustness)",
                    mode=ReportMode.PRINT,
                )
                params[self.key] = new_value

                # Apply cross-dependency validation after mutation
                validator = CrossDependencyPerturbation()
                params = validator.apply(params)

                return params
        # For non-MLP models with solver param, skip mutation to avoid confusion
        return params

    def _handle_learning_rate_protection(self, params: dict, current) -> dict:
        """Handle MLPClassifier learning_rate protection."""
        # MLPClassifier has hidden_layer_sizes (unique identifier)
        if "hidden_layer_sizes" in params:
            safe_options = ["constant", "invscaling", "adaptive"]
            if current not in safe_options:
                new_value = random.choice(safe_options)
                report_data(
                    f"🧪 SCIENTIFIC PROTECTION: MLPClassifier detected, using safe "
                    f"learning_rate '{current}' -> '{new_value}'"
                    f" (preventing InvalidParameterError)",
                    mode=ReportMode.PRINT,
                )
                params[self.key] = new_value
                return params
            # Valid MLP learning_rate, allow mutation within safe options
            choices = [opt for opt in safe_options if opt != current]
            if choices:
                new_value = random.choice(choices)
                report_data(
                    f"🧪 SCIENTIFIC PERTURBATION: Applying MLP learning_rate mutation "
                    f"learning_rate='{current}' -> '{new_value}' (testing robustness)",
                    mode=ReportMode.PRINT,
                )
                params[self.key] = new_value
                return params
        # For non-MLP models with learning_rate param, skip mutation to avoid confusion
        return params

    def apply(self, params: dict) -> dict:
        """Apply string mutation to hyperparameters.

        Args:
            params (dict): Dictionary of model hyperparameters.

        Returns:
            dict: Modified hyperparameters with mutations applied.
        """
        current = params.get(self.key)
        if current is None or not isinstance(current, str):
            return params

        # Apply specific protections
        if self.key == "max_features":
            result = self._handle_max_features_protection(params, current)
            if result is not None:
                return result

        if self.key == "solver":
            return self._handle_solver_protection(params, current)

        if self.key == "learning_rate":
            return self._handle_learning_rate_protection(params, current)

        # Standard mutation logic
        if current in self.options:
            choices = [opt for opt in self.options if opt != current]
            if choices:
                new_value = random.choice(choices)
                params[self.key] = new_value
        else:
            # Scientific interest: apply potentially invalid mutations
            if self.options:
                new_value = random.choice(self.options)
                logger.warning(
                    "🧪 SCIENTIFIC PERTURBATION: Applying potentially invalid mutation "
                    "%s='%s' -> '%s' (testing robustness to invalid parameters)",
                    self.key,
                    current,
                    new_value,
                )
                params[self.key] = new_value

        return params

    @staticmethod
    def supports(value) -> bool:
        """Checks if the value is a string.

        Args:
            value (Any): The value to check.

        Returns:
            bool: True if value is a string, False otherwise.
        """
        return isinstance(value, str)
