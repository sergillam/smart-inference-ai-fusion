"""Benchmark helpers for quantization experiments."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def benchmark_inference(
    predict_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, n_runs: int = 50
) -> float:
    """Measure median inference runtime in milliseconds."""
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0.")

    x_arr = np.asarray(x)
    if x_arr.size == 0:
        raise ValueError("x must be non-empty.")
    timings_ms: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(x_arr)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)
    return float(np.median(timings_ms))


def estimate_memory_bytes(array: np.ndarray) -> int:
    """Estimate memory footprint in bytes for a NumPy array."""
    return int(np.asarray(array).nbytes)


def compute_memory_reduction(baseline_bytes: int, quantized_bytes: int) -> float:
    """Compute relative memory reduction: 1 - (quantized / baseline)."""
    if baseline_bytes <= 0:
        raise ValueError("baseline_bytes must be > 0.")
    if quantized_bytes < 0:
        raise ValueError("quantized_bytes must be >= 0.")
    return float(1.0 - (quantized_bytes / baseline_bytes))


def compute_compression_ratio(baseline_bits: int, quantized_bits: int) -> float:
    """Compute compression ratio baseline_bits / quantized_bits."""
    if baseline_bits <= 0:
        raise ValueError("baseline_bits must be > 0.")
    if quantized_bits <= 0:
        raise ValueError("quantized_bits must be > 0.")
    return float(baseline_bits / quantized_bits)


def compute_overhead_pct(baseline_time_ms: float, quantized_time_ms: float) -> float:
    """Compute runtime overhead percentage for quantized inference."""
    if baseline_time_ms <= 0:
        raise ValueError("baseline_time_ms must be > 0.")
    return float(((quantized_time_ms - baseline_time_ms) / baseline_time_ms) * 100.0)
