"""Parameter transformation for type casting.

Randomly casts numeric or boolean hyperparameters to another type,
such as int→str, float→int, or bool→str.
"""

from .base import ParameterTransformation


class TypeCastPerturbation(ParameterTransformation):
    """Randomly casts numeric or boolean parameters to a different type.

    This transformation is useful for testing model robustness against unexpected
    type changes in configuration or user input. For example:
        - int → str
        - float → int
        - bool → str

    Attributes:
        key (str): The hyperparameter name to cast.
    """

    def __init__(self, key: str):
        """Initializes the TypeCastPerturbation transformation.

        Args:
            key (str): The name of the parameter to cast.
        """
        self.key = key

    def apply(self, params: dict) -> str | int | float | None:
        """Casts the value of the parameter to a different type if possible.

        Args:
            params (dict): Model hyperparameters.

        Returns:
            str | int | float | None: The casted value if applicable, or None if not applicable.
        """
        value = params.get(self.key)
        if isinstance(value, int) and not isinstance(value, bool):  # bool is a subclass of int
            return str(value)  # int → str
        if isinstance(value, float):
            return int(value)  # float → int
        if isinstance(value, bool):
            return str(value)  # bool → str
        return None

    @staticmethod
    def supports(value) -> bool:
        """Checks if the value is eligible for casting.

        Args:
            value (Any): The value to check.

        Returns:
            bool: True if value is int, float, or bool; False otherwise.
        """
        return isinstance(value, (int, float, bool))
