"""Model-parameter quantization utilities for SIP-Q."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from smart_inference_ai_fusion.quantization.core.methods import (
    QUANTIZE_METHODS,
    dequantize,
)


class WeightQuantizer:
    """Quantize trained model parameters using quantize->dequantize roundtrip."""

    def __init__(self, num_bits: int = 8, method: str = "uniform", dtype_profile: str = "integer"):
        if method not in QUANTIZE_METHODS:
            raise ValueError(f"Unsupported method: {method}.")
        if num_bits not in {8, 16, 32}:
            raise ValueError("num_bits must be one of {8, 16, 32}.")
        if dtype_profile not in {"integer", "float16"}:
            raise ValueError("dtype_profile must be 'integer' or 'float16'.")
        if dtype_profile == "float16" and num_bits != 16:
            raise ValueError("dtype_profile='float16' is only supported with num_bits=16.")

        self.num_bits = num_bits
        self.method = method
        self.dtype_profile = dtype_profile

    def quantize_model(self, model: Any) -> Any:
        """Return a quantized deep copy preserving the original input type."""
        estimator = getattr(model, "model", model)
        estimator_copy = copy.deepcopy(estimator)
        self._apply_quantization(estimator_copy)

        if hasattr(model, "model"):
            model_copy = copy.deepcopy(model)
            model_copy.model = estimator_copy
            return model_copy
        return estimator_copy

    def _apply_quantization(self, estimator: Any) -> None:
        if hasattr(estimator, "coefs_"):
            estimator.coefs_ = [self._quantize_roundtrip(weights) for weights in estimator.coefs_]
            estimator.intercepts_ = [
                self._quantize_roundtrip(bias) for bias in estimator.intercepts_
            ]

        if hasattr(estimator, "tree_") and hasattr(estimator.tree_, "threshold"):
            threshold = estimator.tree_.threshold.copy()
            valid_mask = threshold != -2.0  # preserve leaf sentinel
            threshold[valid_mask] = self._quantize_roundtrip(threshold[valid_mask])
            estimator.tree_.threshold[:] = threshold

        fit_x = getattr(estimator, "_fit_X", None)
        if fit_x is not None:
            setattr(estimator, "_fit_X", self._quantize_roundtrip(fit_x))

        if hasattr(estimator, "cluster_centers_"):
            estimator.cluster_centers_ = self._quantize_roundtrip(estimator.cluster_centers_)

        if hasattr(estimator, "means_"):
            estimator.means_ = self._quantize_roundtrip(estimator.means_)

        if hasattr(estimator, "covariances_"):
            estimator.covariances_ = self._quantize_roundtrip(estimator.covariances_)
            self._try_recompute_gmm_precisions(estimator)

    def _quantize_roundtrip(self, arr: np.ndarray) -> np.ndarray:
        source = np.asarray(arr, dtype=np.float64)
        if self.dtype_profile == "float16":
            return source.astype(np.float16).astype(np.float64)

        quantize_fn = QUANTIZE_METHODS[self.method]
        original_shape = source.shape
        flat = source.reshape(-1)
        quantized, params = quantize_fn(flat, num_bits=self.num_bits)
        reconstructed = dequantize(quantized, params).reshape(original_shape)
        return reconstructed.astype(np.float64)

    @staticmethod
    def _try_recompute_gmm_precisions(estimator: Any) -> None:
        if not (hasattr(estimator, "covariances_") and hasattr(estimator, "covariance_type")):
            return
        try:
            estimator.precisions_cholesky_ = _compute_precision_cholesky(
                estimator.covariances_, estimator.covariance_type
            )
        except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):  # pragma: no cover
            # Best-effort recomputation to keep estimator internally consistent.
            return
