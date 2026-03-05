"""Tests for quantization experiment configuration."""

from dataclasses import FrozenInstanceError

import pytest

from smart_inference_ai_fusion.quantization.core.config import QuantizationConfig


def test_config_defaults() -> None:
    """Default values should follow the feature plan."""
    config = QuantizationConfig()
    assert config.data_bits == (8, 16, 32)
    assert config.model_bits == (8, 16, 32)
    assert config.dtype_profile == "integer"
    assert config.method == "uniform"
    assert config.enable_hybrid is True
    assert config.calibration_samples == 1000
    assert config.random_seed == 42


def test_config_is_immutable() -> None:
    """Dataclass must be frozen."""
    config = QuantizationConfig()
    with pytest.raises(FrozenInstanceError):
        config.method = "kmeans"


def test_config_rejects_empty_data_bits() -> None:
    """At least one bit-width must be provided."""
    with pytest.raises(ValueError, match="data_bits cannot be empty"):
        QuantizationConfig(data_bits=())


def test_config_rejects_invalid_model_bits() -> None:
    """Only 8, 16 and 32 are allowed."""
    with pytest.raises(ValueError, match="unsupported bit widths"):
        QuantizationConfig(model_bits=(8, 12))


def test_config_rejects_non_positive_calibration_samples() -> None:
    """Calibration sample size must be positive."""
    with pytest.raises(ValueError, match="calibration_samples must be > 0"):
        QuantizationConfig(calibration_samples=0)


def test_config_warns_when_float16_ignores_method() -> None:
    """float16 profile should warn when method is not uniform."""
    with pytest.warns(UserWarning, match="direct float16 cast"):
        QuantizationConfig(dtype_profile="float16", method="kmeans")


def test_config_rejects_duplicate_data_bits() -> None:
    """Bit-width tuple must not contain duplicates."""
    with pytest.raises(ValueError, match="cannot contain duplicate"):
        QuantizationConfig(data_bits=(8, 8, 16))


def test_config_rejects_negative_random_seed() -> None:
    """Random seed must be non-negative."""
    with pytest.raises(ValueError, match="random_seed must be >= 0"):
        QuantizationConfig(random_seed=-1)
