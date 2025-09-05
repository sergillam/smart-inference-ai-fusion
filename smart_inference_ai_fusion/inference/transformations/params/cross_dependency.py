"""Parameter transformation for cross-dependency perturbation."""

from typing import Any

from smart_inference_ai_fusion.utils.report import ReportMode, report_data

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
        # Model-specific parameter incompatibility rules (remove incompatible params)
        # MLPClassifier incompatible parameters
        {
            "condition": {
                "hidden_layer_sizes": lambda x: x is not None
            },  # MLPClassifier identifier
            "remove": [
                "positive",
                "alpha",
                "fit_intercept",
                "normalize",
                "copy_X",
                "max_iter_inner",
            ],
        },
        # RidgeClassifier incompatible parameters
        {
            "condition": {
                "alpha": lambda x: x is not None,
                "hidden_layer_sizes": lambda x: x is None,
            },  # Ridge identifier
            "remove": [
                "hidden_layer_sizes",
                "activation",
                "learning_rate",
                "learning_rate_init",
                "power_t",
                "max_fun",
                "momentum",
                "nesterovs_momentum",
                "early_stopping",
                "validation_fraction",
                "beta_1",
                "beta_2",
                "epsilon",
                "n_iter_no_change",
            ],
        },
        # Cross-dependency rules (enforce parameter combinations)
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
        # Ridge solver compatibility: lbfgs can only be used when positive=True
        {
            "condition": {"solver": "lbfgs", "positive": False},
            "set": {"solver": "auto"},
        },
        {
            "condition": {"solver": "lbfgs"},
            "set": {"positive": True},
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
            if self._rule_condition_matches(rule["condition"], params):
                params = self._apply_rule(rule, params)
        return params

    def _rule_condition_matches(self, condition: dict, params: dict) -> bool:
        """Check if rule condition matches current parameters.

        Args:
            condition: Rule condition dictionary
            params: Current parameters

        Returns:
            bool: True if condition matches
        """
        for k, v in condition.items():
            param_value = params.get(k)
            if callable(v):  # Lambda function condition
                if not v(param_value):
                    return False
            else:  # Direct value condition
                if param_value != v:
                    return False
        return True

    def _apply_rule(self, rule: dict, params: dict) -> dict:
        """Apply a single rule to parameters.

        Args:
            rule: Rule dictionary containing actions to apply
            params: Current parameters

        Returns:
            dict: Updated parameters
        """
        # Handle parameter removal rules
        if "remove" in rule:
            params = self._apply_removal_rule(rule["remove"], params)

        # Handle parameter setting rules
        if "set" in rule:
            params = self._apply_setting_rule(rule["set"], rule["condition"], params)

        return params

    def _apply_removal_rule(self, params_to_remove: list, params: dict) -> dict:
        """Apply parameter removal rule.

        Args:
            params_to_remove: List of parameter names to remove
            params: Current parameters

        Returns:
            dict: Updated parameters
        """
        for param_to_remove in params_to_remove:
            if param_to_remove in params:
                removed_value = params.pop(param_to_remove)
                report_data(
                    f"🧪 SCIENTIFIC PROTECTION: Removed incompatible parameter "
                    f"{param_to_remove}='{removed_value}' (model incompatibility)",
                    mode=ReportMode.PRINT,
                )
        return params

    def _apply_setting_rule(self, settings: dict, condition: dict, params: dict) -> dict:
        """Apply parameter setting rule.

        Args:
            settings: Dictionary of parameters to set
            condition: Original rule condition for logging
            params: Current parameters

        Returns:
            dict: Updated parameters
        """
        for target_key, target_value in settings.items():
            if params.get(target_key) != target_value:
                old_value = params.get(target_key)
                params[target_key] = target_value

                # Log the scientific protection/dependency enforcement
                condition_str = ", ".join(
                    [f"{k}={v}" for k, v in condition.items() if not callable(v)]
                )
                report_data(
                    f"🧪 SCIENTIFIC DEPENDENCY: When {condition_str}, "
                    f"enforcing {target_key}='{old_value}' -> '{target_value}' "
                    f"(preventing invalid parameter combination)",
                    mode=ReportMode.PRINT,
                )
        return params

    @staticmethod
    def supports(_value: Any) -> bool:
        """Always returns True since this transformation acts on full parameter sets.

        Returns:
            bool: Always True.
        """
        return True
