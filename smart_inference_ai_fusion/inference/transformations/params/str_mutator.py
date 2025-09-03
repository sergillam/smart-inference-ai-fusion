"""Parameter transformation that mutates a string hyperparameter among valid options."""

import logging
import random

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
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
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

    def apply(self, params: dict) -> dict:
        """Mutates the value of the string hyperparameter if present.

        Args:
            params (dict): Model hyperparameters.

        Returns:
            dict: Updated hyperparameters with the mutated value, if applicable.
        """
        if self.key in params and isinstance(params[self.key], str) and self.options:
            current = params[self.key]

            # 🧪 SCIENTIFIC PROTECTION: GradientBoosting max_features compatibility
            if self.key == "max_features":
                # Check if this looks like a GradientBoosting context
                gb_indicators = {"learning_rate", "subsample", "n_estimators"}
                if any(indicator in params for indicator in gb_indicators):
                    # Use GradientBoosting-safe max_features only
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

            # Only mutate if current value is in our known options
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
