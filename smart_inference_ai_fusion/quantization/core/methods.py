"""Quantization methods used by SIP-Q."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

QuantizeOutput = tuple[np.ndarray, dict[str, Any]]
_DTYPE_MAP = {8: np.uint8, 16: np.uint16, 32: np.uint32}
_VALID_BITS = set(_DTYPE_MAP)
_MAX_PERCENTILE_BINS = 4096
_MAX_KMEANS_CLUSTERS = 1024


def _validate_num_bits(num_bits: int) -> None:
    if num_bits not in _VALID_BITS:
        raise ValueError(f"num_bits must be one of {_VALID_BITS}.")


def _validate_non_empty(data: np.ndarray) -> None:
    if data.size == 0:
        raise ValueError("data must be non-empty.")


def _target_dtype(num_bits: int) -> np.dtype:
    return _DTYPE_MAP[num_bits]


def uniform_quantize(data: np.ndarray, num_bits: int = 8) -> QuantizeOutput:
    """Uniform scalar quantization over [min, max]."""
    _validate_num_bits(num_bits)
    arr = np.asarray(data, dtype=np.float64)
    _validate_non_empty(arr)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    dtype = _target_dtype(num_bits)

    if min_val == max_val:
        return np.zeros(arr.shape, dtype=dtype), {
            "method": "uniform",
            "scale": 1.0,
            "zero_point": min_val,
            "min": min_val,
            "max": max_val,
        }

    qmax = (2**num_bits) - 1
    scale = (max_val - min_val) / qmax
    zero_point = min_val
    quantized = np.clip(np.round((arr - zero_point) / scale), 0, qmax).astype(dtype)

    return quantized, {"method": "uniform", "scale": scale, "zero_point": zero_point}


def minmax_quantize(data: np.ndarray, num_bits: int = 8) -> QuantizeOutput:
    """Quantize after min-max normalization to [0, 1]."""
    _validate_num_bits(num_bits)
    arr = np.asarray(data, dtype=np.float64)
    _validate_non_empty(arr)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    dtype = _target_dtype(num_bits)

    if min_val == max_val:
        return np.zeros(arr.shape, dtype=dtype), {
            "method": "minmax",
            "scale": 1.0,
            "zero_point": 0.0,
            "original_min": min_val,
            "original_max": max_val,
        }

    normalized = (arr - min_val) / (max_val - min_val)
    quantized, params = uniform_quantize(normalized, num_bits=num_bits)
    return quantized, {
        "method": "minmax",
        "scale": params["scale"],
        "zero_point": params["zero_point"],
        "original_min": min_val,
        "original_max": max_val,
    }


def kmeans_quantize(data: np.ndarray, num_bits: int = 8) -> QuantizeOutput:
    """Non-uniform quantization through k-means centroids."""
    _validate_num_bits(num_bits)
    arr = np.asarray(data, dtype=np.float64)
    _validate_non_empty(arr)
    flat = arr.reshape(-1, 1)
    dtype = _target_dtype(num_bits)

    unique_vals = np.unique(flat)
    n_clusters = min(2**num_bits, len(unique_vals), _MAX_KMEANS_CLUSTERS)
    if n_clusters <= 1:
        return np.zeros(arr.shape, dtype=dtype), {
            "method": "kmeans",
            "centroids": np.array([float(unique_vals[0])], dtype=np.float64),
        }

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=3,
        batch_size=min(1024, len(flat)),
    )
    labels = kmeans.fit_predict(flat)
    quantized = labels.reshape(arr.shape).astype(dtype)
    return quantized, {"method": "kmeans", "centroids": kmeans.cluster_centers_.reshape(-1)}


def percentile_quantize(data: np.ndarray, num_bits: int = 8) -> QuantizeOutput:
    """Quantize using percentile-based non-uniform bins."""
    _validate_num_bits(num_bits)
    arr = np.asarray(data, dtype=np.float64)
    _validate_non_empty(arr)
    flat = arr.reshape(-1)
    dtype = _target_dtype(num_bits)

    unique_vals = np.unique(flat)
    num_bins = min(2**num_bits, len(unique_vals), _MAX_PERCENTILE_BINS)
    if num_bins <= 1:
        val = float(unique_vals[0])
        bins = np.array([val, val], dtype=np.float64)
        return np.zeros(arr.shape, dtype=dtype), {"method": "percentile", "bins": bins}

    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.unique(np.percentile(flat, percentiles))
    if len(bins) < 2:
        bins = np.array([bins[0], bins[0]], dtype=np.float64)
        return np.zeros(arr.shape, dtype=dtype), {"method": "percentile", "bins": bins}

    quantized = np.digitize(flat, bins, right=False) - 1
    quantized = np.clip(quantized, 0, len(bins) - 2).astype(dtype)
    return quantized.reshape(arr.shape), {"method": "percentile", "bins": bins}


QUANTIZE_METHODS = {
    "uniform": uniform_quantize,
    "minmax": minmax_quantize,
    "kmeans": kmeans_quantize,
    "percentile": percentile_quantize,
}


def dequantize(quantized: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Inverse operation for quantized tensors."""
    q = np.asarray(quantized)
    method = params.get("method", "uniform")

    if method == "uniform":
        scale = float(params["scale"])
        zero_point = float(params["zero_point"])
        return q.astype(np.float64) * scale + zero_point

    if method == "minmax":
        reconstructed_01 = q.astype(np.float64) * float(params["scale"]) + float(
            params["zero_point"]
        )
        original_min = float(params["original_min"])
        original_max = float(params["original_max"])
        return reconstructed_01 * (original_max - original_min) + original_min

    if method == "kmeans":
        centroids = np.asarray(params["centroids"], dtype=np.float64)
        indices = np.clip(q.astype(np.int64), 0, len(centroids) - 1)
        return centroids[indices]

    if method == "percentile":
        bins = np.asarray(params["bins"], dtype=np.float64)
        if len(bins) < 2:
            return np.full(q.shape, bins[0] if len(bins) == 1 else 0.0, dtype=np.float64)
        centers = (bins[:-1] + bins[1:]) / 2.0
        indices = np.clip(q.astype(np.int64), 0, len(centers) - 1)
        return centers[indices]

    raise ValueError(f"Unsupported method for dequantize: {method}")
