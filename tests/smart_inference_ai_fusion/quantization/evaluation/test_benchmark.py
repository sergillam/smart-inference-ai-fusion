"""Tests for quantization benchmark helpers."""

import numpy as np
import pytest

from smart_inference_ai_fusion.quantization.evaluation.benchmark import (
    benchmark_inference,
    compute_compression_ratio,
    compute_memory_reduction,
    compute_overhead_pct,
    estimate_memory_bytes,
)


def test_benchmark_inference_returns_positive_float() -> None:
    """Benchmark should return a non-negative runtime in milliseconds."""

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    x = np.random.default_rng(42).normal(size=(100, 8))
    runtime_ms = benchmark_inference(predict_fn, x, n_runs=5)
    assert isinstance(runtime_ms, float)
    assert runtime_ms >= 0.0


def test_benchmark_inference_rejects_non_positive_runs() -> None:
    """Benchmark should validate the n_runs argument."""

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError, match="n_runs must be > 0"):
        benchmark_inference(predict_fn, np.zeros((3, 2)), n_runs=0)


def test_benchmark_inference_rejects_empty_x() -> None:
    """Benchmark should reject empty input arrays."""

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError, match="x must be non-empty"):
        benchmark_inference(predict_fn, np.empty((0, 2)), n_runs=5)


def test_estimate_memory_bytes_matches_numpy_nbytes() -> None:
    """Memory helper should match NumPy's nbytes value."""
    x = np.zeros((10, 3), dtype=np.float32)
    assert estimate_memory_bytes(x) == x.nbytes


def test_compute_overhead_pct() -> None:
    """Overhead percentage should follow the expected formula."""
    overhead = compute_overhead_pct(10.0, 12.5)
    assert overhead == pytest.approx(25.0)


def test_compute_overhead_pct_rejects_non_positive_baseline() -> None:
    """Overhead computation requires positive baseline runtime."""
    with pytest.raises(ValueError, match="baseline_time_ms must be > 0"):
        compute_overhead_pct(0.0, 2.0)


def test_compute_memory_reduction() -> None:
    """Memory reduction should follow 1 - (quantized / baseline)."""
    reduction = compute_memory_reduction(1000, 250)
    assert reduction == pytest.approx(0.75)


def test_compute_memory_reduction_rejects_invalid_inputs() -> None:
    """Memory reduction should validate byte inputs."""
    with pytest.raises(ValueError, match="baseline_bytes must be > 0"):
        compute_memory_reduction(0, 10)
    with pytest.raises(ValueError, match="quantized_bytes must be >= 0"):
        compute_memory_reduction(100, -1)


def test_compute_compression_ratio() -> None:
    """Compression ratio should be baseline_bits / quantized_bits."""
    ratio = compute_compression_ratio(64, 8)
    assert ratio == pytest.approx(8.0)


def test_compute_compression_ratio_rejects_invalid_inputs() -> None:
    """Compression ratio should validate bit-width inputs."""
    with pytest.raises(ValueError, match="baseline_bits must be > 0"):
        compute_compression_ratio(0, 8)
    with pytest.raises(ValueError, match="quantized_bits must be > 0"):
        compute_compression_ratio(64, 0)
