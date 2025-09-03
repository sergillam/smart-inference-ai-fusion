"""Parameter transformation for bounded numeric shift."""

import logging
import random

from .base import ParameterTransformation

logger = logging.getLogger(__name__)


class BoundedNumericShift(ParameterTransformation):
    """Applies an additive perturbation to a float hyperparameter within a defined bound.

    This transformation is useful for continuous hyperparameters such as
    'alpha', 'gamma', 'learning_rate', 'var_smoothing', etc.

    Attributes:
        key (str): Name of the parameter to be perturbed.
        shift_range (tuple): Minimum and maximum shift values.
    """

    def __init__(self, key: str, shift_range: tuple = (-0.1, 0.1)):
        """Initializes the BoundedNumericShift transformation.

        Args:
            key (str): Name of the parameter to be perturbed.
            shift_range (tuple, optional): Minimum and maximum shift values.
                Defaults to (-0.1, 0.1).
        """
        self.key = key
        self.shift_range = shift_range

    def apply(self, params: dict) -> dict:
        """Applies the shift to the specified float parameter in-place.

        Args:
            params (dict): Dictionary of model hyperparameters.

        Returns:
            dict: Updated parameter dictionary with the shifted value.
        """
        params = params.copy()
        value = params.get(self.key)

        if isinstance(value, float):
            # 🧪 SCIENTIFIC PROTECTION: Strict parameter ranges for critical parameters
            strict_param_ranges = {
                "subsample": (
                    0.1,
                    1.0,
                ),  # GradientBoosting subsample ∈ (0.0, 1.0], conservative range
            }

            if self.key in strict_param_ranges:
                min_val, max_val = strict_param_ranges[self.key]

                # Calculate very conservative shift range
                max_positive_shift = min(0.05, max_val - value)  # Maximum 5% increase
                max_negative_shift = min(0.05, value - min_val)  # Maximum 5% decrease

                # Limit shift to very conservative range
                safe_shift_min = max(self.shift_range[0], -max_negative_shift)
                safe_shift_max = min(self.shift_range[1], max_positive_shift)

                if safe_shift_min >= safe_shift_max:
                    # No safe shift possible, skip transformation to avoid any issues
                    logger.warning(
                        "🧪 SCIENTIFIC PROTECTION: No safe shift for %s=%.3f "
                        "(strict range: %.1f-%.1f), skipping to prevent fatal error",
                        self.key,
                        value,
                        min_val,
                        max_val,
                    )
                    return params

                # Apply very conservative shift
                shift = random.uniform(safe_shift_min, safe_shift_max)
                new_value = value + shift

                # Double-check bounds (extra safety)
                new_value = max(min_val, min(max_val, new_value))

                logger.debug(
                    "🧪 SCIENTIFIC SHIFT: %s %.3f -> %.3f (conservative shift: %.3f)",
                    self.key,
                    value,
                    new_value,
                    shift,
                )

                params[self.key] = new_value
            else:
                # Original logic unchanged for all other parameters (no side effects)
                shift = random.uniform(*self.shift_range)
                params[self.key] = max(0.0, value + shift)  # Avoid negative values

        return params

    @staticmethod
    def supports(value) -> bool:
        """Checks if the value is a float.

        Args:
            value (Any): Value to check.

        Returns:
            bool: True if value is a float, False otherwise.
        """
        return isinstance(value, float)
