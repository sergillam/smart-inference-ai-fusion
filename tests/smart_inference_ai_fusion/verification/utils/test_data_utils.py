"""Tests for verification shared utilities."""

import numpy as np

from smart_inference_ai_fusion.verification.utils.data_utils import (
    build_class_balance_metrics,
    check_data_shape_validation,
    check_parameter_initialization,
    check_precondition_data_preprocessing,
    normalize_to_array,
    parse_noise_test_params,
    parse_shape_config,
    parse_type_safety_config,
    verify_classification_constraints,
    verify_probability_bounds,
)


def test_normalize_to_array_from_scalar():
    """Scalar inputs should become 1-element arrays."""
    arr = normalize_to_array(7)
    assert arr.shape == (1,)
    assert arr[0] == 7


def test_normalize_to_array_from_nested_list():
    """Nested lists should be flattened."""
    arr = normalize_to_array([[1, 2], [3, 4]])
    assert arr.tolist() == [1, 2, 3, 4]


def test_parse_shape_config_defaults():
    """Default shape config values should be stable."""
    expected_input, expected_output, preserve_batch = parse_shape_config({})
    assert expected_input is None
    assert expected_output is None
    assert preserve_batch is True


def test_parse_type_safety_config_custom_values():
    """Type safety parser should return explicit values from config."""
    expected_type, allow_none = parse_type_safety_config(
        {"expected_type": "integer", "allow_none": True}
    )
    assert expected_type == "integer"
    assert allow_none is True


def test_parse_noise_test_params_values():
    """Noise test parser should extract configured thresholds."""
    noise_level, stability_threshold = parse_noise_test_params(
        {"noise_level": 0.2, "stability_threshold": 0.1}
    )
    assert noise_level == 0.2
    assert stability_threshold == 0.1


def test_verify_probability_bounds_detects_violations():
    """Probability bounds checker should flag values outside [0, 1]."""
    is_valid, detail = verify_probability_bounds(np.array([-0.1, 0.6, 1.1]))
    assert is_valid is False
    assert detail["type"] == "probability_bounds"


def test_verify_classification_constraints_detects_out_of_range():
    """Class constraints checker should flag invalid class ids."""
    is_valid, detail = verify_classification_constraints(np.array([0, 1, 3]), 3)
    assert is_valid is False
    assert detail["type"] == "classification_constraints"


def test_check_parameter_initialization_not_found():
    """Missing required parameters should return violation metadata."""
    ok, error_type, detail = check_parameter_initialization("lr", {"epochs": 10})
    assert ok is False
    assert error_type == "not_found"
    assert detail["param_name"] == "lr"


def test_check_data_shape_validation_mismatch():
    """Shape validation should report expected and actual shapes."""
    data = np.zeros((2, 3))
    ok, detail = check_data_shape_validation(data, [2, 2])
    assert ok is False
    assert detail["type"] == "data_shape_validation"


def test_check_precondition_data_preprocessing_violation():
    """Highly unnormalized data should trigger preprocessing violation."""
    data = np.array([0] * 19 + [1000], dtype=float)
    ok, detail = check_precondition_data_preprocessing(data)
    assert ok is False
    assert detail["type"] == "data_preprocessing_violation"


def test_build_class_balance_metrics_empty_array():
    """Empty labels should return zero-safe metrics."""
    metrics = build_class_balance_metrics(np.array([]))
    assert metrics["unique_classes"] == 0
    assert metrics["total_samples"] == 0


def test_verify_probability_bounds_success_valid_range():
    """✅ SUCCESS: Valid probability arrays should pass bounds verification."""
    is_valid, detail = verify_probability_bounds(np.array([0.0, 0.5, 1.0]))
    assert is_valid is True
    assert detail == {}


def test_verify_classification_constraints_failure_invalid_classes():
    """❌ FAILURE: Classification constraints should fail with invalid class ids."""
    is_valid, detail = verify_classification_constraints(np.array([0, 1, 5]), 3)
    assert is_valid is False
    assert detail["invalid_high_count"] > 0
