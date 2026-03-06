"""Feature quantization utilities for dataset inputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from smart_inference_ai_fusion.quantization.core.methods import (
    QUANTIZE_METHODS,
    dequantize,
)

DTYPE_MAP = {8: np.uint8, 16: np.uint16, 32: np.uint32}


class FeatureQuantizer:
    """Quantize tabular features column-wise."""

    def __init__(self, method: str = "uniform", num_bits: int = 8, dtype_profile: str = "integer"):
        if method not in QUANTIZE_METHODS:
            raise ValueError(f"Unsupported method: {method}.")
        if num_bits not in DTYPE_MAP:
            raise ValueError(f"num_bits must be one of {set(DTYPE_MAP)}.")
        if dtype_profile not in {"integer", "float16"}:
            raise ValueError("dtype_profile must be 'integer' or 'float16'.")
        if dtype_profile == "float16" and num_bits != 16:
            raise ValueError("dtype_profile='float16' is only supported with num_bits=16.")

        self.method = method
        self.num_bits = num_bits
        self.dtype_profile = dtype_profile
        self._params: list[dict[str, Any]] = []
        self._fitted = False

    def fit(self, x: np.ndarray) -> "FeatureQuantizer":
        """Calibrate quantization parameters feature by feature."""
        x_arr = self._ensure_2d(x)
        if self.dtype_profile == "float16":
            self._params = [{"method": "float16_cast"} for _ in range(x_arr.shape[1])]
            self._fitted = True
            return self

        quantize_fn = QUANTIZE_METHODS[self.method]
        self._params = []
        for idx in range(x_arr.shape[1]):
            _, params = quantize_fn(x_arr[:, idx], num_bits=self.num_bits)
            self._params.append(params)
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Quantize features using previously calibrated parameters."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        x_arr = self._ensure_2d(x)
        self._validate_feature_count(x_arr)
        if self.dtype_profile == "float16":
            return x_arr.astype(np.float16)

        out = np.zeros(x_arr.shape, dtype=DTYPE_MAP[self.num_bits])
        for idx, params in enumerate(self._params):
            out[:, idx] = self._quantize_with_params(x_arr[:, idx], params, self.num_bits)
        return out

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit and transform in a single step."""
        return self.fit(x).transform(x)

    def inverse_transform(self, x_quantized: np.ndarray) -> np.ndarray:
        """Reconstruct approximate float64 representation."""
        x_q = self._ensure_2d(x_quantized)

        if not self._fitted:
            raise RuntimeError("Call fit() before inverse_transform().")
        self._validate_feature_count(x_q)

        if self.dtype_profile == "float16":
            return x_q.astype(np.float64)

        reconstructed = np.zeros(x_q.shape, dtype=np.float64)
        for idx, params in enumerate(self._params):
            reconstructed[:, idx] = dequantize(x_q[:, idx], params)
        return reconstructed

    @staticmethod
    def _ensure_2d(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            return x_arr.reshape(-1, 1)
        if x_arr.ndim != 2:
            raise ValueError("FeatureQuantizer expects a 1D or 2D array.")
        return x_arr

    def _validate_feature_count(self, x_arr: np.ndarray) -> None:
        expected = len(self._params)
        if expected != x_arr.shape[1]:
            raise ValueError(f"Expected {expected} features based on fit(), got {x_arr.shape[1]}.")

    @staticmethod
    def _quantize_with_params(
        column: np.ndarray, params: dict[str, Any], num_bits: int
    ) -> np.ndarray:
        method = params["method"]
        col = np.asarray(column, dtype=np.float64)
        qmax = (2**num_bits) - 1

        if method == "uniform":
            scale = float(params["scale"])
            zero_point = float(params["zero_point"])
            if scale == 0:
                return np.zeros(col.shape, dtype=np.int64)
            return np.clip(np.round((col - zero_point) / scale), 0, qmax).astype(np.int64)

        if method == "minmax":
            original_min = float(params["original_min"])
            original_max = float(params["original_max"])
            if original_max == original_min:
                return np.zeros(col.shape, dtype=np.int64)
            normalized = (col - original_min) / (original_max - original_min)
            scale = float(params["scale"])
            zero_point = float(params["zero_point"])
            return np.clip(np.round((normalized - zero_point) / scale), 0, qmax).astype(np.int64)

        if method == "kmeans":
            centroids = np.asarray(params["centroids"], dtype=np.float64).reshape(-1, 1)
            distances = np.abs(col.reshape(-1, 1) - centroids.T)
            return np.argmin(distances, axis=1).astype(np.int64)

        if method == "percentile":
            bins = np.asarray(params["bins"], dtype=np.float64)
            indices = np.digitize(col, bins, right=False) - 1
            return np.clip(indices, 0, len(bins) - 2).astype(np.int64)

        raise ValueError(f"Unsupported method in params: {method}")
