"""Parameter transformation for injecting random integer noise."""

import random
import logging
logger = logging.getLogger(__name__)
from .base import ParameterTransformation


class IntegerNoise(ParameterTransformation):
    """Applies small random integer noise to an integer hyperparameter.

    This transformation perturbs the given parameter by adding or subtracting
    a random integer within the specified delta range.

    Attributes:
        key (str): Name of the hyperparameter to perturb.
        delta (int): Maximum absolute value of noise to add or subtract.
    """

    def __init__(self, key: str, delta: int = 2):
        """Initializes the IntegerNoise transformation.

        Args:
            key (str): Name of the hyperparameter to perturb.
            delta (int, optional): Range for random perturbation (default is 2).
        """
        self.key = key
        self.delta = delta

    def apply(self, params: dict) -> dict:
        """Applies integer noise to the selected parameter if it is an integer.

        Args:
            params (dict): Dictionary of model hyperparameters.

        Returns:
            dict: Updated dictionary with the perturbed integer parameter.
        """
        
        params = params.copy()
        
        # 🧪 SCIENTIFIC PROTECTION: Detect float-like parameter ranges
        FLOAT_RANGE_PARAMS = {
            "subsample": (0.0, 1.0),  # GradientBoosting subsample must be in (0.0, 1.0]
            "learning_rate": (0.0, 1.0),  # Learning rates typically in (0.0, 1.0]
            "alpha": (0.0, float('inf')),  # Regularization parameters must be positive
        }
        
        if self.key in params and isinstance(params[self.key], (int, float)):
            original_value = params[self.key]
            
            # Check if this is a float parameter that should not get integer noise
            if self.key in FLOAT_RANGE_PARAMS:
                min_val, max_val = FLOAT_RANGE_PARAMS[self.key]
                logger.warning(
                    "🧪 SCIENTIFIC PROTECTION: Parameter %s=%.3f is float-range parameter, "
                    "skipping integer noise (valid range: %.1f-%.1f)",
                    self.key, original_value, min_val, max_val
                )
                # Ensure it's a proper float and within valid range
                if isinstance(original_value, int):
                    # Convert int back to float if needed
                    float_value = float(original_value)
                    # Clamp to valid range
                    clamped_value = max(min_val + 0.001, min(max_val, float_value))
                    if float_value != clamped_value:
                        logger.warning(
                            "🧪 SCIENTIFIC PROTECTION: Clamping %s from %.3f to %.3f",
                            self.key, float_value, clamped_value
                        )
                    params[self.key] = clamped_value
                return params
            
            # Apply integer noise only to actual integer parameters
            if isinstance(original_value, int):
                new_value = original_value + random.randint(-self.delta, self.delta)
                params[self.key] = new_value
                
        return params

    @staticmethod
    def supports(value) -> bool:
        """Checks if the value is an integer.

        Args:
            value (Any): Value to check.

        Returns:
            bool: True if value is an integer.
        """
        return isinstance(value, int)
