"""Parameter transformation that flips a boolean parameter value."""

from .base import ParameterTransformation

class BooleanFlip(ParameterTransformation):
    """Flips the value of a boolean parameter.

    This transformation negates the value of a specified boolean key in the parameter dictionary.

    Attributes:
        key (str): The parameter name to be flipped.
    """

    def __init__(self, key: str):
        """Initializes the BooleanFlip transformation.

        Args:
            key (str): The parameter name to be flipped.
        """
        self.key = key

    def apply(self, params: dict) -> dict:
        """Flips the boolean value of the specified parameter.

        Args:
            params (dict): Dictionary of model parameters.

        Returns:
            dict: Dictionary with the specified boolean parameter flipped.
        """
        if self.key in params and isinstance(params[self.key], bool):
            params[self.key] = not params[self.key]
        return params

    @staticmethod
    def supports(value) -> bool:
        """Checks if the value is a boolean type.

        Args:
            value (Any): Value to check.

        Returns:
            bool: True if the value is a boolean, False otherwise.
        """
        return isinstance(value, bool)
