"""Common utilities for model wrappers."""

from typing import Any, Callable, Dict, Optional


def initialize_model_params(params: Optional[dict], **kwargs: Any) -> dict:
    """Initialize model parameters with optional defaults.

    Args:
        params: Optional dictionary of parameters.
        **kwargs: Additional keyword arguments.

    Returns:
        Merged parameter dictionary.
    """
    if params is None:
        params = {}
    params.update(kwargs)
    return params


def create_base_metrics_dict() -> Dict[str, Optional[float]]:
    """Create base metrics dictionary with None values.

    Returns:
        Dictionary with standard metric keys initialized to None.
    """
    return {
        "accuracy": None,
        "balanced_accuracy": None,
        "f1": None,
    }


def safe_metric_calculation(calculation_func: Callable[[], Any], fallback_value: Any = None) -> Any:
    """Safely execute a metric calculation with a fallback value.

    Args:
        calculation_func (Callable[[], Any]): The function to execute for the calculation.
        fallback_value (Any): The value to return if the calculation fails.

    Returns:
        Any: The result of the calculation or the fallback value.
    """
    try:
        return calculation_func()
    except (ValueError, TypeError):
        return fallback_value
