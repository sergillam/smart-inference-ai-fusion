"""Smoke tests for quantization package imports."""

from smart_inference_ai_fusion.quantization import (
    BitWidth,
    DTypeProfile,
    FeatureQuantizer,
    QuantizationConfig,
    QuantizationResult,
    QuantMethod,
    WeightQuantizer,
    dequantize,
    kmeans_quantize,
    minmax_quantize,
    percentile_quantize,
    uniform_quantize,
)


def test_public_api_imports() -> None:
    """Top-level imports should expose planned public symbols."""
    assert QuantizationConfig is not None
    assert QuantizationResult is not None
    assert BitWidth is not None
    assert DTypeProfile is not None
    assert QuantMethod is not None
    assert FeatureQuantizer is not None
    assert WeightQuantizer is not None
    assert uniform_quantize is not None
    assert minmax_quantize is not None
    assert kmeans_quantize is not None
    assert percentile_quantize is not None
    assert dequantize is not None
