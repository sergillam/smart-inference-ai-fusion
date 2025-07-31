"""Parameter transformation for simulating boundary shifts in enum hyperparameters."""

import random
from .base import ParameterTransformation

class EnumBoundaryShift(ParameterTransformation):
    """Simulates a boundary shift in ordered enum-like categorical hyperparameters.

    For example, if 'weights' is 'uniform' and valid values are ['uniform', 'distance'],
    this transformation flips it to the adjacent value in the order.
    
    Attributes:
        ENUM_ORDERED_SPACE (dict): Ordered lists for supported categorical keys.
    """

    ENUM_ORDERED_SPACE = {
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "penalty": ["l2", "l1", "elasticnet", "none"],
    }

    def __init__(self, key: str):
        """Initializes the EnumBoundaryShift transformation.

        Args:
            key (str): Name of the hyperparameter to perturb.
        """
        self.key = key

    def apply(self, params: dict) -> dict:
        """Flips the enum value to an adjacent boundary value if possible.

        Args:
            params (dict): Model hyperparameter dictionary.

        Returns:
            dict: Updated dictionary with the value shifted, if possible; otherwise unchanged.
        """
        params = params.copy()
        value = params.get(self.key)
        if value is None:
            return params

        enum_list = self.ENUM_ORDERED_SPACE.get(self.key)
        if enum_list and value in enum_list:
            idx = enum_list.index(value)
            neighbors = []
            if idx > 0:
                neighbors.append(enum_list[idx - 1])
            if idx < len(enum_list) - 1:
                neighbors.append(enum_list[idx + 1])
            if neighbors:
                params[self.key] = random.choice(neighbors)
        return params

    @staticmethod
    def supports(value) -> bool:
        """Checks if the value is a string (suitable for enum perturbation).

        Args:
            value (Any): Value to check.

        Returns:
            bool: True if value is a string.
        """
        return isinstance(value, str)
