"""Parameter transformation that mutates a string hyperparameter among valid options."""

import random

from .base import ParameterTransformation


class StringMutator(ParameterTransformation):
    """Randomly mutates a string hyperparameter among provided options.

    Useful for categorical hyperparameters, e.g. {"gini", "entropy", "log_loss"}.

    Attributes:
        key (str): The hyperparameter name to mutate.
        options (list): List of valid options to choose from.
    """

    def __init__(self, key: str, options=None):
        """Initializes the StringMutator transformation.

        Args:
            key (str): The hyperparameter name to mutate.
            options (list, optional): List of possible string values. Defaults to
                ["gini", "entropy", "log_loss"].
        """
        self.key = key
        self.options = options or ["gini", "entropy", "log_loss"]

    def apply(self, params: dict) -> dict:
        """Mutates the value of the string hyperparameter if present.

        Args:
            params (dict): Model hyperparameters.

        Returns:
            dict: Updated hyperparameters with the mutated value, if applicable.
        """
        if self.key in params and isinstance(params[self.key], str):
            current = params[self.key]
            choices = [opt for opt in self.options if opt != current]
            if choices:
                params[self.key] = random.choice(choices)
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
