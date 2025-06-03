import random
from .base import ParameterTransformation

class EnumBoundaryShift(ParameterTransformation):
    """
    Simulates boundary shift in ordered enum-like categorical hyperparameters.

    Example:
        If "weights" = "uniform" and valid values are ["uniform", "distance"],
        it flips to the adjacent value in list order.
    """

    ENUM_ORDERED_SPACE = {
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "penalty": ["l2", "l1", "elasticnet", "none"],
    }

    def __init__(self, key: str):
        self.key = key

    def apply(self, params: dict) -> str | None:
        value = params.get(self.key)
        if value is None:
            return None

        enum_list = self.ENUM_ORDERED_SPACE.get(self.key)
        if enum_list and value in enum_list:
            idx = enum_list.index(value)
            neighbors = []
            if idx > 0:
                neighbors.append(enum_list[idx - 1])
            if idx < len(enum_list) - 1:
                neighbors.append(enum_list[idx + 1])
            if neighbors:
                return random.choice(neighbors)
        return None

    @staticmethod
    def supports(value) -> bool:
        return isinstance(value, str)
