"""Smoke tests for quantization package imports."""

from smart_inference_ai_fusion.quantization import (
    BitWidth,
    DTypeProfile,
    QuantizationConfig,
    QuantizationResult,
    QuantMethod,
)


def test_public_api_imports() -> None:
    """Top-level imports should expose planned public symbols."""
    assert QuantizationConfig is not None
    assert QuantizationResult is not None
    assert BitWidth is not None
    assert DTypeProfile is not None
    assert QuantMethod is not None
