"""Shared data utilities for verification plugins.

This module contains common functions used by both CVC5 and Z3 plugins
to avoid code duplication (R0801 lint errors).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def normalize_to_array(data: Any) -> np.ndarray:
    """Normalize data to a flattened numpy array.

    Used by both CVC5 and Z3 plugins for consistent data handling.

    Args:
        data: Input data (array, list, scalar, or iterable)

    Returns:
        Flattened numpy array
    """
    if hasattr(data, "__iter__") and not isinstance(data, (str, dict)):
        return np.array(data).flatten()
    return np.array([data]).flatten()


def extract_numeric_data(obj: Any) -> Optional[np.ndarray]:
    """Extract numeric data recursively from dicts/arrays.

    Searches for common keys like 'train', 'test', 'data', 'values', etc.

    Args:
        obj: Input object (dict, array, or other)

    Returns:
        Numpy array if found, None otherwise
    """
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        for key in ["train", "test", "data", "values", "input", "output"]:
            if key in obj:
                result = extract_numeric_data(obj[key])
                if result is not None:
                    return result
        return None
    if hasattr(obj, "__iter__") and not isinstance(obj, str):
        return np.array(obj)
    return None


def check_value_finite(value: Any) -> Tuple[bool, str]:
    """Check if a value is finite (not NaN or Inf).

    Args:
        value: Value to check

    Returns:
        Tuple of (is_finite, error_description)
    """
    try:
        float_val = float(value)
        if not np.isfinite(float_val):
            if np.isnan(float_val):
                return False, "nan_value"
            return False, "inf_value"
        return True, ""
    except (TypeError, ValueError):
        return False, "conversion_error"


def try_convert_to_float(value: Any) -> Tuple[Optional[float], bool]:
    """Try to convert value to float and check if finite.

    Args:
        value: Value to convert

    Returns:
        Tuple of (float_value or None, is_finite)
        - If conversion fails, returns (None, False)
        - If NaN/Inf, returns (float_val, False)
        - If valid finite, returns (float_val, True)
    """
    try:
        float_val = float(value)
        if not np.isfinite(float_val):
            return float_val, False
        return float_val, True
    except (TypeError, ValueError):
        return None, False


def check_type_numeric(value: Any) -> bool:
    """Check if value is a numeric type.

    Args:
        value: Value to check

    Returns:
        True if numeric, False otherwise
    """
    return isinstance(value, (int, float, np.integer, np.floating))


def check_type_integer(value: Any, tolerance: float = 1e-9) -> bool:
    """Check if value is an integer or close to integer.

    Args:
        value: Value to check
        tolerance: Tolerance for float-to-integer conversion

    Returns:
        True if integer-like, False otherwise
    """
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isclose(value, round(value), atol=tolerance))
    return False


def check_strict_integer(float_val: float, tolerance: float = 1e-9) -> Tuple[bool, float, float]:
    """Check if a float value is close to an integer.

    Args:
        float_val: Float value to check
        tolerance: Tolerance for integer check

    Returns:
        Tuple of (is_integer, rounded_value, difference)
    """
    rounded = round(float_val)
    diff = abs(float_val - rounded)
    is_int = diff <= tolerance
    return is_int, rounded, diff


def check_type_boolean(value: Any) -> bool:
    """Check if value is a boolean type.

    Args:
        value: Value to check

    Returns:
        True if boolean, False otherwise
    """
    return isinstance(value, (bool, np.bool_))


def verify_type_safety(
    data_array: np.ndarray,
    expected_type: str = "numeric",
    allow_none: bool = False,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Verify type safety for all values in array.

    Common implementation for CVC5 and Z3 plugins.

    Args:
        data_array: Flattened numpy array of values
        expected_type: Expected type ('numeric', 'integer', 'boolean')
        allow_none: Whether None values are allowed

    Returns:
        Tuple of (all_satisfied, list of violation examples)
    """
    all_satisfied = True
    violations = []

    for i, value in enumerate(data_array):
        try:
            # Check None values
            if value is None:
                if not allow_none:
                    all_satisfied = False
                    violations.append(
                        {
                            "type": "none_value",
                            "index": int(i),
                            "explanation": f"None value at index {i}, but allow_none=False",
                        }
                    )
                continue

            # Check type based on expected_type
            if expected_type == "numeric":
                if not check_type_numeric(value):
                    all_satisfied = False
                    violations.append(
                        {
                            "type": "type_mismatch",
                            "index": int(i),
                            "expected_type": expected_type,
                            "actual_type": type(value).__name__,
                            "explanation": (
                                f"Expected numeric, got {type(value).__name__} " f"at index {i}"
                            ),
                        }
                    )

            elif expected_type == "integer":
                if not isinstance(value, (int, np.integer)):
                    if isinstance(value, (float, np.floating)):
                        if not float(value).is_integer():
                            all_satisfied = False
                            violations.append(
                                {
                                    "type": "type_mismatch",
                                    "index": int(i),
                                    "expected_type": expected_type,
                                    "actual_value": float(value),
                                    "explanation": (
                                        f"Expected integer, got float {value} " f"at index {i}"
                                    ),
                                }
                            )
                    else:
                        all_satisfied = False
                        violations.append(
                            {
                                "type": "type_mismatch",
                                "index": int(i),
                                "expected_type": expected_type,
                                "actual_type": type(value).__name__,
                                "explanation": (
                                    f"Expected integer, got {type(value).__name__} " f"at index {i}"
                                ),
                            }
                        )

            elif expected_type == "boolean":
                if not check_type_boolean(value):
                    all_satisfied = False
                    violations.append(
                        {
                            "type": "type_mismatch",
                            "index": int(i),
                            "expected_type": expected_type,
                            "actual_type": type(value).__name__,
                            "explanation": (
                                f"Expected boolean, got {type(value).__name__} " f"at index {i}"
                            ),
                        }
                    )

        except Exception:
            pass

    return all_satisfied, violations[:10]


def verify_shape_preservation(
    input_data: Optional[np.ndarray],
    output_data: Optional[np.ndarray],
    expected_input_shape: Optional[Tuple[int, ...]] = None,
    expected_output_shape: Optional[Tuple[int, ...]] = None,
    preserve_batch_dim: bool = True,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Verify shape preservation constraints.

    Common implementation for CVC5 and Z3 plugins.

    Args:
        input_data: Input data array
        output_data: Output data array
        expected_input_shape: Expected shape of input
        expected_output_shape: Expected shape of output
        preserve_batch_dim: Whether batch dimension should be preserved

    Returns:
        Tuple of (all_satisfied, list of violation examples)
    """
    all_satisfied = True
    violations = []

    # Check input shape
    if input_data is not None and expected_input_shape is not None:
        input_array = np.array(input_data)
        actual_shape = input_array.shape
        expected = tuple(expected_input_shape)

        if actual_shape != expected:
            all_satisfied = False
            violations.append(
                {
                    "type": "input_shape_mismatch",
                    "actual_shape": list(actual_shape),
                    "expected_shape": list(expected),
                    "explanation": f"Input shape {actual_shape} != expected {expected}",
                }
            )

    # Check output shape
    if output_data is not None and expected_output_shape is not None:
        output_array = np.array(output_data)
        actual_shape = output_array.shape
        expected = tuple(expected_output_shape)

        if actual_shape != expected:
            all_satisfied = False
            violations.append(
                {
                    "type": "output_shape_mismatch",
                    "actual_shape": list(actual_shape),
                    "expected_shape": list(expected),
                    "explanation": f"Output shape {actual_shape} != expected {expected}",
                }
            )

    # Check batch dimension preservation
    if preserve_batch_dim and input_data is not None and output_data is not None:
        input_array = np.array(input_data)
        output_array = np.array(output_data)

        if len(input_array.shape) > 0 and len(output_array.shape) > 0:
            if input_array.shape[0] != output_array.shape[0]:
                all_satisfied = False
                violations.append(
                    {
                        "type": "batch_dim_mismatch",
                        "input_batch_dim": int(input_array.shape[0]),
                        "output_batch_dim": int(output_array.shape[0]),
                        "explanation": (
                            f"Batch dimension not preserved: "
                            f"input={input_array.shape[0]}, output={output_array.shape[0]}"
                        ),
                    }
                )

    return all_satisfied, violations


def verify_finite_values(
    data_array: np.ndarray,
    allow_nan: bool = False,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Verify all values are finite (no NaN or Inf).

    Common implementation for CVC5 and Z3 plugins.

    Args:
        data_array: Flattened numpy array of values
        allow_nan: Whether NaN values are allowed

    Returns:
        Tuple of (all_satisfied, list of violation examples)
    """
    all_satisfied = True
    violations = []

    for i, value in enumerate(data_array):
        try:
            float_val = float(value)

            # Check NaN
            if np.isnan(float_val):
                if not allow_nan:
                    all_satisfied = False
                    violations.append(
                        {
                            "type": "nan_value",
                            "index": int(i),
                            "value": "NaN",
                            "explanation": "NaN value found but allow_nan=False",
                        }
                    )
                continue

            # Check Inf
            if np.isinf(float_val):
                all_satisfied = False
                violations.append(
                    {
                        "type": "inf_value",
                        "index": int(i),
                        "value": str(float_val),
                        "explanation": f"Infinite value found: {float_val}",
                    }
                )

        except (TypeError, ValueError) as e:
            all_satisfied = False
            violations.append(
                {
                    "type": "conversion_error",
                    "index": int(i),
                    "explanation": f"Cannot convert to float: {e}",
                }
            )

    return all_satisfied, violations[:10]


def verify_classification_constraints(
    data_array: np.ndarray,
    num_classes: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Verify classification output constraints.

    Checks that predicted classes are in valid range [0, num_classes).

    Args:
        data_array: Array of predicted classes
        num_classes: Number of valid classes

    Returns:
        Tuple of (is_valid, violation details dict)
    """
    invalid_low = np.sum(data_array < 0)
    invalid_high = np.sum(data_array >= num_classes)

    if invalid_low > 0 or invalid_high > 0:
        return False, {
            "type": "classification_constraints",
            "num_classes": num_classes,
            "invalid_low_count": int(invalid_low),
            "invalid_high_count": int(invalid_high),
            "explanation": (
                f"Classes outside [0, {num_classes}): "
                f"{invalid_low} below 0, {invalid_high} >= {num_classes}"
            ),
        }

    return True, {}


def verify_probability_bounds(
    data_array: np.ndarray,
) -> Tuple[bool, Dict[str, Any]]:
    """Verify probability bounds constraints.

    Checks that all probabilities are in range [0, 1].

    Args:
        data_array: Array of probability values

    Returns:
        Tuple of (is_valid, violation details dict)
    """
    below_zero = np.sum(data_array < 0)
    above_one = np.sum(data_array > 1)

    if below_zero > 0 or above_one > 0:
        return False, {
            "type": "probability_bounds",
            "below_zero_count": int(below_zero),
            "above_one_count": int(above_one),
            "min_value": float(np.min(data_array)),
            "max_value": float(np.max(data_array)),
            "explanation": (
                f"Probabilities outside [0,1]: " f"{below_zero} below 0, {above_one} above 1"
            ),
        }

    return True, {}


def get_data_from_input(
    input_data: Any,
    constraint_value: Any = None,
    prefer_output: bool = False,
) -> Optional[np.ndarray]:
    """Extract data from input_data object or constraint_value.

    Common pattern used in verification plugins.

    Args:
        input_data: VerificationInput object with input_data/output_data
        constraint_value: Fallback constraint value dict
        prefer_output: Whether to prefer output_data over input_data

    Returns:
        Numpy array of data, or None if not found
    """
    data = None

    if input_data is not None:
        if prefer_output:
            if hasattr(input_data, "output_data") and input_data.output_data is not None:
                data = extract_numeric_data(input_data.output_data)
            if (
                data is None
                and hasattr(input_data, "input_data")
                and input_data.input_data is not None
            ):
                data = extract_numeric_data(input_data.input_data)
        else:
            if hasattr(input_data, "input_data") and input_data.input_data is not None:
                data = extract_numeric_data(input_data.input_data)
            if (
                data is None
                and hasattr(input_data, "output_data")
                and input_data.output_data is not None
            ):
                data = extract_numeric_data(input_data.output_data)

    if data is None and constraint_value is not None:
        if isinstance(constraint_value, dict):
            data = constraint_value.get("data", None)
            if data is not None:
                data = np.array(data)

    return data


def parse_shape_config(
    constraint_data: Any,
) -> Tuple[Optional[Tuple], Optional[Tuple], bool]:
    """Parse shape configuration from constraint data.

    Args:
        constraint_data: Constraint configuration dict or other

    Returns:
        Tuple of (expected_input_shape, expected_output_shape, preserve_batch_dim)
    """
    if isinstance(constraint_data, dict):
        expected_input_shape = constraint_data.get("expected_input_shape", None)
        expected_output_shape = constraint_data.get("expected_output_shape", None)
        preserve_batch_dim = constraint_data.get("preserve_batch_dim", True)
    else:
        expected_input_shape = None
        expected_output_shape = None
        preserve_batch_dim = True

    return expected_input_shape, expected_output_shape, preserve_batch_dim


def parse_type_safety_config(
    constraint_data: Any,
) -> Tuple[str, bool]:
    """Parse type safety configuration from constraint data.

    Args:
        constraint_data: Constraint configuration dict or other

    Returns:
        Tuple of (expected_type, allow_none)
    """
    if isinstance(constraint_data, dict):
        expected_type = constraint_data.get("expected_type", "numeric")
        allow_none = constraint_data.get("allow_none", False)
    else:
        expected_type = "numeric"
        allow_none = False

    return expected_type, allow_none


def parse_robustness_tests(constraint_data: Any) -> List[Dict[str, Any]]:
    """Parse robustness tests from constraint data.

    Args:
        constraint_data: Constraint configuration dict

    Returns:
        List of robustness test configurations
    """
    if not isinstance(constraint_data, dict):
        return []
    return constraint_data.get("tests", [])


def parse_adversarial_test_params(test: Dict[str, Any]) -> Tuple[float, str, float]:
    """Parse adversarial robustness test parameters.

    Args:
        test: Test configuration dict

    Returns:
        Tuple of (epsilon, norm_type, output_threshold)
    """
    epsilon = test.get("epsilon", 0.1)
    norm_type = test.get("norm", "l2")
    output_threshold = test.get("output_threshold", 0.1)
    return epsilon, norm_type, output_threshold


def parse_noise_test_params(test: Dict[str, Any]) -> Tuple[float, float]:
    """Parse noise robustness test parameters.

    Args:
        test: Test configuration dict

    Returns:
        Tuple of (noise_level, stability_threshold)
    """
    noise_level = test.get("noise_level", 0.1)
    stability_threshold = test.get("stability_threshold", 0.05)
    return noise_level, stability_threshold


def get_robustness_test_type(test: Dict[str, Any]) -> str:
    """Get the type of a robustness test.

    Args:
        test: Test configuration dict

    Returns:
        Test type string
    """
    return test.get("type", "")


# === FUNÇÕES PARA CONVERSÃO DE RESULTADOS ===

# Mapeamento de status legado para padronizado (constante compartilhada)
LEGACY_STATUS_MAPPING: Dict[str, str] = {
    "SUCCESS": "SUCCESS",
    "FAILURE": "FAILURE",
    "ERROR": "ERROR",
    "SKIPPED": "SKIPPED",
    "TIMEOUT": "TIMEOUT",
}


def compute_avg_time_per_constraint(execution_time: float, constraints_count: int) -> float:
    """Compute average time per constraint.

    Args:
        execution_time: Total execution time
        constraints_count: Number of constraints

    Returns:
        Average time per constraint
    """
    return execution_time / max(constraints_count, 1)


def build_performance_metrics_dict(
    execution_time: float,
    constraints_checked: List[Any],
    constraints_satisfied: List[Any],
    constraints_violated: List[Any],
) -> Dict[str, Any]:
    """Build performance metrics dictionary from legacy result data.

    Args:
        execution_time: Total execution time
        constraints_checked: List of checked constraints
        constraints_satisfied: List of satisfied constraints
        constraints_violated: List of violated constraints

    Returns:
        Dictionary with performance metrics
    """
    return {
        "total_execution_time": execution_time,
        "constraint_count": len(constraints_checked),
        "constraints_satisfied": len(constraints_satisfied),
        "constraints_violated": len(constraints_violated),
        "constraints_unknown": 0,
        "constraints_timeout": 0,
        "constraints_error": 0,
        "constraints_skipped": 0,
    }


def get_output_data_array(input_data: Any) -> Optional[np.ndarray]:
    """Extract output data from input_data object as flattened array.

    Common pattern for postcondition verification in plugins.

    Args:
        input_data: VerificationInput object with output_data

    Returns:
        Flattened numpy array of output data, or None if not available
    """
    if input_data is None:
        return None

    data = getattr(input_data, "output_data", None)
    if data is None:
        return None

    if not hasattr(data, "__iter__"):
        return None

    return np.array(data).flatten()


def check_postcondition_classification(
    input_data: Any, num_classes: int
) -> Tuple[bool, Dict[str, Any]]:
    """Check classification postcondition on output data.

    Args:
        input_data: VerificationInput object with output_data
        num_classes: Number of valid classes

    Returns:
        Tuple of (is_valid, violation_detail_dict)
    """
    data_array = get_output_data_array(input_data)
    if data_array is None:
        return True, {}  # No data = trivially satisfied

    return verify_classification_constraints(data_array, num_classes)


def check_nan_value_in_bounds(
    value: float, index: int, allow_nan: bool
) -> Tuple[bool, bool, Optional[Dict[str, Any]]]:
    """Check if value is NaN and handle accordingly for bounds verification.

    Args:
        value: The value to check
        index: Index of the value in the array
        allow_nan: Whether NaN values are allowed

    Returns:
        Tuple of (should_skip, failed, violation_detail or None)
        - should_skip: True if this value should be skipped (continue in loop)
        - failed: True if this value causes a violation
        - violation_detail: Details about the violation, or None
    """
    if np.isnan(value):
        if not allow_nan:
            return (
                True,
                True,
                {
                    "type": "nan_value",
                    "index": int(index),
                    "value": "NaN",
                    "explanation": "NaN value found but allow_nan=False",
                },
            )
        return True, False, None  # Skip but don't fail
    return False, False, None  # Don't skip


def check_value_non_negative(value: float, index: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check if value is non-negative after conversion.

    Args:
        value: The float value to check
        index: Index of the value in the array

    Returns:
        Tuple of (is_valid, violation_detail or None)
    """
    if value < 0:
        return False, {
            "type": "negative_value",
            "index": int(index),
            "value": value,
            "explanation": f"Value {value} is negative",
        }
    return True, None


def check_value_finite_for_non_negative(value: Any) -> Tuple[bool, bool]:
    """Check if value is finite (not NaN or Inf) for non-negative verification.

    Args:
        value: Value to check

    Returns:
        Tuple of (is_finite, should_fail)
        - is_finite: False if NaN or Inf
        - should_fail: True if this should cause a validation failure
    """
    if np.isnan(value) or np.isinf(value):
        return False, True
    return True, False


def handle_constraint_verification_error(
    constraint_type: str, error_result: Dict[str, Any], log_func=None
) -> None:
    """Handle a constraint verification error with optional fallback.

    Common pattern for both CVC5 and Z3 plugins when handling errors.

    Args:
        constraint_type: Type of constraint that failed
        error_result: Error handling result dict from error handler
        log_func: Optional logger function (e.g., logger.info)
    """
    # Apply fallback if suggested by error handler
    if error_result.get("action") == "use_basic_constraints":
        if log_func:
            log_func(f"🔧 Applying basic fallback for {constraint_type}")
        # Continuar com constraints simplificados


# Default verification constraints configuration
DEFAULT_VERIFICATION_CONSTRAINTS: Dict[str, bool] = {
    "shape_preservation": True,
    "bounds": True,
    "range_check": True,
    "type_safety": True,
}


def get_default_verification_constraints(strict: bool = False) -> Dict[str, Any]:
    """Get default verification constraints configuration.

    Args:
        strict: If True, use stricter tolerance values.

    Returns:
        Dictionary with default constraint settings including bounds_tolerance.
    """
    constraints = DEFAULT_VERIFICATION_CONSTRAINTS.copy()
    constraints["bounds_tolerance"] = 0.05 if strict else 0.1
    return constraints


# === SHARED REPORTING AND VERIFICATION UTILITIES ===


def build_verification_session_dict(
    verifier_name: str,
    timestamp: str,
    execution_time: float,
    total_constraints: int,
    constraints_satisfied: int,
    constraints_violated: int,
) -> Dict[str, Any]:
    """Build verification session dictionary for reports.

    Common structure used by both CVC5 and Z3 plugins.

    Args:
        verifier_name: Name of the verifier (e.g., "cvc5", "z3")
        timestamp: Timestamp string
        execution_time: Execution time in seconds
        total_constraints: Total number of constraints
        constraints_satisfied: Number of satisfied constraints
        constraints_violated: Number of violated constraints

    Returns:
        Dictionary with verification session data
    """
    return {
        "verifier": verifier_name,
        "timestamp": timestamp,
        "execution_time_ms": round(execution_time * 1000, 2),
        "total_constraints": total_constraints,
        "constraints_satisfied": constraints_satisfied,
        "constraints_violated": constraints_violated,
        "success_rate": round(constraints_satisfied / max(1, total_constraints) * 100, 1),
    }


def build_error_context_dict(
    constraints: List[str],
    execution_time: float,
    timeout: float,
    logic: Optional[str] = None,
) -> Dict[str, Any]:
    """Build error context dictionary for error handling.

    Common structure used by both CVC5 and Z3 plugins.

    Args:
        constraints: List of constraint keys
        execution_time: Execution time in seconds
        timeout: Timeout value
        logic: Optional logic type (e.g., "QF_NRA" for CVC5)

    Returns:
        Dictionary with error context
    """
    context = {
        "constraints": constraints,
        "execution_time": execution_time,
        "timeout": timeout,
    }
    if logic:
        context["logic"] = logic
    return context


def log_constraint_violation(
    constraint: str,
    details: Dict[str, Any],
    log_func,
    get_category_func,
) -> None:
    """Log a constraint violation with counterexample details.

    Common logging pattern used by both CVC5 and Z3 plugins.

    Args:
        constraint: Name of the violated constraint
        details: Details dict containing counterexample info
        log_func: Logger function (e.g., logger.info)
        get_category_func: Function to get constraint category
    """
    counterexample = details.get("counterexample", {})

    # Cabeçalho do constraint violado
    log_func("")
    log_func("🚨 VIOLAÇÃO: [%s]", constraint.upper())
    log_func("   Categoria: %s", get_category_func(constraint))

    if counterexample:
        violation_examples = counterexample.get("violation_examples", [])
        violation_count = counterexample.get("violation_count", len(violation_examples))

        log_func("   Total de violações: %d", violation_count)
        log_func("   📝 CONTRAEXEMPLOS:")

        for i, example in enumerate(violation_examples[:5], 1):
            example_type = example.get("type", "unknown")
            explanation = example.get("explanation", "Sem explicação")

            log_func("      [%d] Tipo: %s", i, example_type)
            log_func("          Detalhe: %s", explanation)


def check_data_consistency(
    data: Any,
) -> Tuple[bool, int, int]:
    """Check data consistency (no NaN/Inf values).

    Common invariant check used by both CVC5 and Z3 plugins.

    Args:
        data: Data to check (array-like)

    Returns:
        Tuple of (is_consistent, nan_count, inf_count)
    """
    if not hasattr(data, "__iter__") or isinstance(data, str):
        return True, 0, 0

    data_array = np.array(data).flatten()
    nan_count = int(np.sum(np.isnan(data_array)))
    inf_count = int(np.sum(np.isinf(data_array)))

    is_consistent = nan_count == 0 and inf_count == 0
    return is_consistent, nan_count, inf_count


def check_parameter_validity(
    param_name: str,
    value: Any,
    min_val: float,
    max_val: float,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check if a parameter value is within bounds.

    Common invariant check used by both CVC5 and Z3 plugins.

    Args:
        param_name: Name of the parameter
        value: Parameter value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Tuple of (is_valid, violation_detail or None)
    """
    if min_val <= value <= max_val:
        return True, None

    return False, {
        "type": "parameter_validity",
        "param_name": param_name,
        "value": value,
        "expected_min": min_val,
        "expected_max": max_val,
        "explanation": f"Parameter {param_name}={value} outside bounds [{min_val}, {max_val}]",
    }


def check_data_shape_validation(
    data: Any,
    expected_shape: Optional[List[int]],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check if data shape matches expected shape.

    Common precondition check used by both CVC5 and Z3 plugins.

    Args:
        data: Data with shape attribute
        expected_shape: Expected shape as list

    Returns:
        Tuple of (is_valid, violation_detail or None)
    """
    if expected_shape is None or not hasattr(data, "shape"):
        return True, None

    actual_shape = data.shape
    if actual_shape == tuple(expected_shape):
        return True, None

    return False, {
        "type": "data_shape_validation",
        "expected_shape": expected_shape,
        "actual_shape": list(actual_shape),
        "explanation": f"Shape mismatch: expected {expected_shape}, got {actual_shape}",
    }


# === SHARED DATA EXTRACTION UTILITIES ===


def extract_input_output_data(input_data: Any) -> Tuple[Any, Any]:
    """Extract input_data and output_data from VerificationInput object.

    Common pattern used by both CVC5 and Z3 plugins.

    Args:
        input_data: VerificationInput object

    Returns:
        Tuple of (input_d, output_d) - both may be None
    """
    input_d = None
    output_d = None

    if input_data and hasattr(input_data, "input_data") and input_data.input_data is not None:
        input_d = input_data.input_data
    if input_data and hasattr(input_data, "output_data") and input_data.output_data is not None:
        output_d = input_data.output_data

    return input_d, output_d


def check_output_validity(
    output_data: Any,
) -> Tuple[bool, int, int]:
    """Check if output data is valid (no NaN/Inf).

    Common postcondition check used by both CVC5 and Z3 plugins.

    Args:
        output_data: Output data to check

    Returns:
        Tuple of (is_valid, nan_count, inf_count)
    """
    if output_data is None:
        return True, 0, 0

    if not hasattr(output_data, "__iter__") or isinstance(output_data, str):
        return True, 0, 0

    data_array = np.array(output_data).flatten()
    nan_count = int(np.sum(np.isnan(data_array)))
    inf_count = int(np.sum(np.isinf(data_array)))

    is_valid = nan_count == 0 and inf_count == 0
    return is_valid, nan_count, inf_count


def check_parameter_initialization(
    param_name: str,
    parameters: Dict[str, Any],
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Check if a required parameter is initialized.

    Common precondition check used by both CVC5 and Z3 plugins.

    Args:
        param_name: Name of required parameter
        parameters: Parameters dict

    Returns:
        Tuple of (is_valid, error_type or None, violation_detail or None)
        - error_type: "not_found" or "is_none" if invalid
    """
    if param_name not in parameters:
        return (
            False,
            "not_found",
            {
                "type": "parameter_initialization",
                "param_name": param_name,
                "explanation": f"Required parameter '{param_name}' not found",
            },
        )

    if parameters[param_name] is None:
        return (
            False,
            "is_none",
            {
                "type": "parameter_initialization",
                "param_name": param_name,
                "explanation": f"Parameter '{param_name}' is None",
            },
        )

    return True, None, None


def log_verification_summary(
    log_func,
    verifier_name: str,
    session_name: str,
    execution_time: float,
    satisfied_count: int,
    violated_count: int,
    total_count: int,
    success_rate: float,
) -> None:
    """Log verification summary to console.

    Common logging pattern used by both CVC5 and Z3 plugins.

    Args:
        log_func: Logger function (e.g., logger.info)
        verifier_name: Name of the verifier (e.g., "CVC5", "Z3")
        session_name: Name/timestamp of the session
        execution_time: Execution time in seconds
        satisfied_count: Number of satisfied constraints
        violated_count: Number of violated constraints
        total_count: Total number of constraints
        success_rate: Success rate percentage
    """
    log_func("=" * 70)
    log_func("🔬 %s FORMAL VERIFICATION REPORT - %s", verifier_name, session_name)
    log_func("=" * 70)
    log_func("⏱️  Tempo de Execução: %.2fms", execution_time * 1000)
    log_func(
        "📊 Resultado: %d ✅ SATISFEITOS | %d ❌ VIOLADOS | %d total",
        satisfied_count,
        violated_count,
        total_count,
    )
    log_func("📈 Taxa de Sucesso: %.1f%%", success_rate)
    log_func("-" * 70)


def check_nan_for_bounds(
    value: Any, index: int, allow_nan: bool
) -> Tuple[bool, bool, Optional[Dict[str, Any]]]:
    """Check for NaN in bounds verification context.

    Used by both CVC5 and Z3 for consistent NaN handling in bounds verification.

    Args:
        value: The value to check
        index: Index in the array
        allow_nan: Whether NaN values are allowed

    Returns:
        Tuple of (should_continue, is_violated, violation_detail or None)
    """
    if np.isnan(value):
        if not allow_nan:
            return (
                True,
                True,
                {
                    "type": "nan_value",
                    "index": int(index),
                    "value": "NaN",
                    "explanation": "NaN value found but allow_nan=False",
                },
            )
        return True, False, None
    return False, False, None


def check_inf_for_bounds(value: Any, index: int) -> Tuple[bool, bool, Optional[Dict[str, Any]]]:
    """Check for Inf in bounds verification context.

    Used by both CVC5 and Z3 for consistent Inf handling in bounds verification.

    Args:
        value: The value to check
        index: Index in the array

    Returns:
        Tuple of (should_continue, is_violated, violation_detail or None)
    """
    if np.isinf(value):
        return (
            True,
            True,
            {
                "type": "inf_value",
                "index": int(index),
                "value": str(value),
                "explanation": "Infinite value found",
            },
        )
    return False, False, None


def check_non_negative_for_constraint(
    value: Any, index: int
) -> Tuple[bool, bool, Optional[Dict[str, Any]]]:
    """Check if value is non-negative for constraint checking.

    Used by both CVC5 and Z3 for consistent non-negative checking.

    Args:
        value: The value to check
        index: Index in the array

    Returns:
        Tuple of (is_valid, should_continue, violation_detail or None)
    """
    is_finite = np.isfinite(value)
    if not is_finite:
        return False, True, None

    float_val = float(value)
    if float_val < 0:
        return (
            False,
            True,
            {
                "type": "negative_value",
                "index": int(index),
                "value": float_val,
                "explanation": f"Value {float_val} is negative",
            },
        )

    return True, False, None


def build_constraint_result(
    constraint_name: str,
    status: str,
    execution_time: float,
    solver_specific_details: Dict[str, Any],
    error_message: Optional[str] = None,
):
    """Build a ConstraintResult object for satisfied or violated constraint.

    Common pattern used by both CVC5 and Z3 plugins.

    Args:
        constraint_name: Name of the constraint
        status: Status (e.g., "SUCCESS" or "FAILURE")
        execution_time: Execution time for this constraint
        solver_specific_details: Solver-specific details dict
        error_message: Optional error message for failures

    Returns:
        ConstraintResult object
    """
    # Import here to avoid circular imports
    from ..core.result_schema import ConstraintResult, StandardStatus, StandardVerificationResult

    status_enum = StandardStatus.SUCCESS if status == "SUCCESS" else StandardStatus.FAILURE

    return ConstraintResult(
        constraint_type=StandardVerificationResult._classify_constraint_type(constraint_name),
        constraint_name=constraint_name,
        status=status_enum,
        execution_time=execution_time,
        solver_specific_details=solver_specific_details,
        error_message=error_message,
    )


def build_bulk_constraint_results(
    legacy_result: Any,
    avg_time_per_constraint: float,
    solver_details_key: str = "cvc5_solver_details",
) -> List[Any]:
    """Build constraint results from legacy verification result.

    Handles satisfied and violated constraints.
    Common pattern used by both CVC5 and Z3 plugins to avoid code duplication.

    Args:
        legacy_result: Legacy VerificationResult from solver
        avg_time_per_constraint: Average execution time per constraint
        solver_details_key: Key for solver details in legacy_result.details

    Returns:
        List of ConstraintResult objects
    """
    constraint_results = []

    # Constraints satisfeitas
    for constraint_name in legacy_result.constraints_satisfied:
        constraint_results.append(
            build_constraint_result(
                constraint_name,
                "SUCCESS",
                avg_time_per_constraint,
                legacy_result.details.get(solver_details_key, {}).get(constraint_name, {}),
            )
        )

    # Constraints violadas
    for constraint_name in legacy_result.constraints_violated:
        constraint_results.append(
            build_constraint_result(
                constraint_name,
                "FAILURE",
                avg_time_per_constraint,
                legacy_result.details.get(solver_details_key, {}).get(constraint_name, {}),
                error_message="Constraint violated",
            )
        )

    return constraint_results


def build_solver_performance_and_status(legacy_result: Any) -> Tuple[Any, Any]:
    """Build performance metrics and overall status from legacy result.

    Common pattern used by both CVC5 and Z3 plugins to avoid code duplication.

    Args:
        legacy_result: Legacy VerificationResult from solver

    Returns:
        Tuple of (PerformanceMetrics, overall_status)
    """
    from ..core.result_schema import PerformanceMetrics, StandardStatus, VerificationStatus

    # Métricas de performance
    perf_dict = build_performance_metrics_dict(
        legacy_result.execution_time,
        legacy_result.constraints_checked,
        legacy_result.constraints_satisfied,
        legacy_result.constraints_violated,
    )
    performance = PerformanceMetrics(**perf_dict)

    # Converter status legado para padronizado
    status_mapping = {
        VerificationStatus.SUCCESS: StandardStatus.SUCCESS,
        VerificationStatus.FAILURE: StandardStatus.FAILURE,
        VerificationStatus.ERROR: StandardStatus.ERROR,
        VerificationStatus.SKIPPED: StandardStatus.SKIPPED,
        VerificationStatus.TIMEOUT: StandardStatus.TIMEOUT,
    }

    overall_status = status_mapping.get(legacy_result.status, StandardStatus.UNKNOWN)

    return performance, overall_status


def build_verification_report_dict(
    verifier_name: str,
    input_data_name: str,
    execution_time: float,
    total_constraints: int,
    satisfied_count: int,
    violated_count: int,
    solver_details: Dict[str, Any],
    solver_details_key: str = "solver_details",
) -> Dict[str, Any]:
    """Build verification report dictionary for both CVC5 and Z3 plugins.

    Args:
        verifier_name: Name of the verifier
        input_data_name: Name of the input data/session
        execution_time: Total execution time
        total_constraints: Total number of constraints
        satisfied_count: Number of satisfied constraints
        violated_count: Number of violated constraints
        solver_details: Solver-specific details dict
        solver_details_key: Key name for solver details
            (e.g., "cvc5_solver_details", "z3_solver_details")

    Returns:
        Dictionary with verification report
    """
    return {
        "verification_session": build_verification_session_dict(
            verifier_name,
            input_data_name,
            execution_time,
            total_constraints,
            satisfied_count,
            violated_count,
        ),
        "constraint_results": {
            "satisfied": [],
            "violated": [],
        },
        solver_details_key: solver_details,
    }


def log_all_constraint_violations(
    violations: List[Tuple[str, Dict[str, Any]]],
    log_func,
    get_category_func,
) -> None:
    """Log all constraint violations with counterexample details.

    Common logging pattern used by both CVC5 and Z3 plugins.

    Args:
        violations: List of (constraint_name, details) tuples to log
        log_func: Logger function (e.g., logger.info)
        get_category_func: Function to get constraint category
    """
    for constraint_name, details in violations:
        counterexample = details.get("counterexample", {})

        # Cabeçalho do constraint violado
        log_func("")
        log_func("🚨 VIOLAÇÃO: [%s]", constraint_name.upper())
        log_func("   Categoria: %s", get_category_func(constraint_name))

        if counterexample:
            violation_examples = counterexample.get("violation_examples", [])
            violation_count = counterexample.get("violation_count", len(violation_examples))

            log_func("   Total de violações: %d", violation_count)
            log_func("   📝 CONTRAEXEMPLOS:")

            for i, example in enumerate(violation_examples[:5], 1):
                example_type = example.get("type", "unknown")
                explanation = example.get("explanation", "Sem explicação")

                log_func("      [%d] Tipo: %s", i, example_type)
                log_func("          Detalhe: %s", explanation)

                # Mostrar valores específicos
                if "value" in example:
                    log_func("          Valor: %s", example["value"])
                if "index" in example:
                    log_func("          Índice: %d", example["index"])


def check_parameter_validity_for_invariant(
    param_name: str,
    parameters: Dict[str, Any],
    bounds: Dict[str, float],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check parameter validity for invariant verification.

    Common pattern used by both CVC5 and Z3 plugins for parameter_validity invariants.

    Args:
        param_name: Name of the parameter to check
        parameters: Parameters dict
        bounds: Bounds dict with "min" and "max" keys

    Returns:
        Tuple of (is_valid, violation_detail or None)
    """
    if param_name not in parameters:
        return True, None

    value = parameters[param_name]
    min_val = bounds.get("min", float("-inf"))
    max_val = bounds.get("max", float("inf"))

    return check_parameter_validity(param_name, value, min_val, max_val)


def extract_data_for_verification(
    input_data: Any, constraint_type: str = "generic", constraint_value: Any = None
) -> Tuple[Optional[Any], bool]:
    """Extract input or output data for constraint verification.

    Common pattern used by both CVC5 and Z3 plugins to get data from VerificationInput.

    Args:
        input_data: VerificationInput object
        constraint_type: Type of constraint to help determine which data to extract
        constraint_value: Constraint configuration (unused, for API compatibility)

    Returns:
        Tuple of (data or None, has_data: bool)
    """
    if input_data is None:
        return None, False

    # Try input_data first
    if hasattr(input_data, "input_data") and input_data.input_data is not None:
        return input_data.input_data, True

    # Fall back to output_data
    if hasattr(input_data, "output_data") and input_data.output_data is not None:
        return input_data.output_data, True

    return None, False


def build_class_balance_metrics(y: np.ndarray) -> Dict[str, Any]:
    """Build class balance metrics from label array.

    Common pattern used by experiments.common and label_runner for computing label statistics.

    Args:
        y: Array of class labels

    Returns:
        Dictionary with unique_classes, total_samples, class_counts,
        class_fractions, and class_balance
    """
    y_arr = np.asarray(y)
    total_samples = len(y_arr)

    # Handle empty label arrays to avoid division by zero
    if total_samples == 0:
        return {
            "unique_classes": 0,
            "total_samples": 0,
            "class_counts": {},
            "class_fractions": {},
            "class_balance": {
                "min_class_fraction": 0.0,
                "max_class_fraction": 0.0,
                "imbalance_ratio": 0.0,
            },
        }

    unique, counts = np.unique(y_arr, return_counts=True)

    return {
        "unique_classes": len(unique),
        "total_samples": int(total_samples),
        "class_counts": {str(cls): int(cnt) for cls, cnt in zip(unique, counts)},
        "class_fractions": {
            str(cls): float(cnt / total_samples) for cls, cnt in zip(unique, counts)
        },
        "class_balance": {
            "min_class_fraction": float(counts.min() / total_samples) if len(counts) > 0 else 0.0,
            "max_class_fraction": float(counts.max() / total_samples) if len(counts) > 0 else 0.0,
            "imbalance_ratio": (
                float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
            ),
        },
    }


def check_precondition_data_preprocessing(
    data: Any, skip_normalization_check: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check precondition: data should be preprocessed (normalized).

    Common pattern used by cvc5_plugin and z3_plugin for verifying data preprocessing.

    Args:
        data: Data array to check
        skip_normalization_check: Whether to skip normalization validation

    Returns:
        Tuple of (is_satisfied, violation_detail or None)
    """
    if not hasattr(data, "__iter__") or isinstance(data, str):
        return True, None

    data_array = np.array(data).flatten()

    # Verificar normalização (dados entre -3 e 3 desvios padrão)
    if len(data_array) > 1:
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        if std_val > 0:
            normalized = (data_array - mean_val) / std_val
            out_of_range = np.sum(np.abs(normalized) > 3)

            if out_of_range > 0 and not skip_normalization_check:
                return (
                    False,
                    {
                        "type": "data_preprocessing_violation",
                        "issue": "Data not properly normalized",
                        "details": f"{out_of_range} values outside [-3, 3] std range",
                        "mean": float(mean_val),
                        "std": float(std_val),
                    },
                )

    return True, None
