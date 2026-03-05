"""Tests for quantization result schema."""

import pytest
from pydantic import ValidationError

from smart_inference_ai_fusion.quantization.core.result import QuantizationResult


def _build_result() -> QuantizationResult:
    return QuantizationResult(
        experiment_type="data_quant",
        dataset_name="Wine",
        algorithm_name="KNN",
        quantization_method="uniform",
        bit_width=8,
        baseline_accuracy=0.95,
        quantized_accuracy=0.90,
        accuracy_degradation=0.05,
        baseline_memory_bytes=11392,
        quantized_memory_bytes=1424,
        compression_ratio=8.0,
        baseline_time_ms=1.2,
        quantized_time_ms=1.3,
        overhead_pct=8.33,
        quantization_mse=0.003,
        seed=42,
    )


def test_result_serialization_contains_dataset() -> None:
    """Serialized JSON should include key identity fields."""
    result = _build_result()
    payload = result.model_dump_json()
    assert '"dataset_name":"Wine"' in payload
    assert '"algorithm_name":"KNN"' in payload


def test_result_defaults_dtype_and_metadata() -> None:
    """Optional defaults should be filled automatically."""
    result = _build_result()
    assert result.dtype_profile == "integer"
    assert result.metadata == {}


def test_result_dict_roundtrip_preserves_values() -> None:
    """Pydantic roundtrip should preserve numeric fields."""
    original = _build_result()
    rebuilt = QuantizationResult.model_validate(original.model_dump())
    assert rebuilt.quantization_mse == original.quantization_mse
    assert rebuilt.quantized_memory_bytes == original.quantized_memory_bytes


def test_result_rejects_invalid_bit_width() -> None:
    """Bit width must be one of 8, 16, 32."""
    payload = _build_result().model_dump()
    payload["bit_width"] = 7
    with pytest.raises(ValidationError):
        QuantizationResult.model_validate(payload)


def test_result_rejects_quantized_memory_greater_than_baseline() -> None:
    """Quantized payload cannot increase memory for this schema contract."""
    payload = _build_result().model_dump()
    payload["quantized_memory_bytes"] = payload["baseline_memory_bytes"] + 1
    with pytest.raises(ValidationError, match="cannot exceed"):
        QuantizationResult.model_validate(payload)


def test_result_rejects_negative_mse() -> None:
    """Quantization error metric cannot be negative."""
    payload = _build_result().model_dump()
    payload["quantization_mse"] = -0.1
    with pytest.raises(ValidationError):
        QuantizationResult.model_validate(payload)
