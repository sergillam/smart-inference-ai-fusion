"""Tests for quantization core methods."""

import numpy as np
import pytest

from smart_inference_ai_fusion.quantization.core.methods import (
    dequantize,
    kmeans_quantize,
    minmax_quantize,
    percentile_quantize,
    uniform_quantize,
)


def test_uniform_int8_range() -> None:
    """8-bit uniform quantization should stay inside [0, 255]."""
    x = np.linspace(-3.0, 2.0, 200, dtype=np.float64)
    quantized, _ = uniform_quantize(x, num_bits=8)
    assert quantized.dtype == np.uint8
    assert int(quantized.min()) >= 0
    assert int(quantized.max()) <= 255


def test_uniform_reconstruction_error_decreases_with_bits() -> None:
    """16-bit roundtrip should be more accurate than 8-bit."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, 2000)

    q8, p8 = uniform_quantize(x, num_bits=8)
    q16, p16 = uniform_quantize(x, num_bits=16)

    mse8 = np.mean((x - dequantize(q8, p8)) ** 2)
    mse16 = np.mean((x - dequantize(q16, p16)) ** 2)
    assert mse16 < mse8


def test_uniform_constant_input_returns_zero_bins() -> None:
    """Constant arrays should map to a single quantized level."""
    x = np.ones(100, dtype=np.float64) * 7.0
    quantized, params = uniform_quantize(x, num_bits=16)
    reconstructed = dequantize(quantized, params)
    assert np.all(quantized == 0)
    assert np.allclose(reconstructed, 7.0)


def test_minmax_roundtrip_returns_float64() -> None:
    """Min-max path should dequantize to float64 with low error for 16-bit."""
    x = np.linspace(10.0, 20.0, 300, dtype=np.float64)
    quantized, params = minmax_quantize(x, num_bits=16)
    reconstructed = dequantize(quantized, params)
    assert quantized.dtype == np.uint16
    assert reconstructed.dtype == np.float64
    assert np.mean((x - reconstructed) ** 2) < 1e-4


def test_kmeans_quantize_outputs_labels_and_centroids() -> None:
    """K-means quantization should expose centroids for inverse mapping."""
    x = np.array([0.0, 0.1, 0.2, 10.0, 10.1, 9.9], dtype=np.float64)
    quantized, params = kmeans_quantize(x, num_bits=8)
    reconstructed = dequantize(quantized, params)
    assert quantized.shape == x.shape
    assert "centroids" in params
    assert reconstructed.shape == x.shape


def test_percentile_quantize_dtype_and_shape() -> None:
    """Percentile quantizer should preserve shape and target dtype."""
    rng = np.random.default_rng(1)
    x = rng.normal(5.0, 2.0, (50, 3))
    quantized, params = percentile_quantize(x, num_bits=8)
    reconstructed = dequantize(quantized, params)
    assert quantized.dtype == np.uint8
    assert quantized.shape == x.shape
    assert reconstructed.shape == x.shape


def test_methods_reject_invalid_bit_width() -> None:
    """Only 8, 16 and 32-bit modes are supported."""
    x = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="num_bits"):
        uniform_quantize(x, num_bits=12)


def test_methods_reject_empty_arrays() -> None:
    """Quantizers should fail fast on empty arrays."""
    x = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="non-empty"):
        uniform_quantize(x, num_bits=8)
    with pytest.raises(ValueError, match="non-empty"):
        minmax_quantize(x, num_bits=8)
