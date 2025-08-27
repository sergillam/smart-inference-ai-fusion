"""Parameter transformation for cross-dependency perturbation."""

from typing import Any
from .base import ParameterTransformation


class CrossDependencyPerturbation(ParameterTransformation):
    """Alters dependent hyperparameters together based on predefined rules.

    This transformation enforces cross-parameter constraints, such as:
    if 'penalty' == 'l1', then 'solver' must be 'liblinear'.

    Attributes:
        RULES (list): List of rules specifying parameter dependencies.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the CrossDependencyPerturbation transformation."""
        # No initialization needed, but method kept for extensibility.

    RULES = [
        {
            "condition": {"penalty": "l1"},
            "set": {"solver": "liblinear"},
        },
        {
            "condition": {"penalty": "l2"},
            "set": {"solver": "lbfgs"},
        },
        {
            "condition": {"kernel": "linear"},
            "set": {"gamma": "auto"},
        },
        # Add more rules as needed.
    ]

    def apply(self, params: dict) -> dict:
        """Applies cross-dependency rules to the parameter dictionary.

        Args:
            params (dict): Model hyperparameters.

        Returns:
            dict: Updated hyperparameter dictionary after applying dependency rules.
        """
        params = params.copy()
        for rule in self.RULES:
            condition = rule["condition"]
            if all(params.get(k) == v for k, v in condition.items()):
                for target_key, target_value in rule["set"].items():
                    if params.get(target_key) != target_value:
                        params[target_key] = target_value
        return params

    @staticmethod
    def supports(_value: Any) -> bool:
        """Always returns True since this transformation acts on full parameter sets.

        Returns:
            bool: Always True.
        """
        return True
