"""Quantization method interfaces for SIP-Q.

This module defines public function signatures used in Phase 2.
Implementations are intentionally deferred.
"""

from typing import Any

import numpy as np

QuantizeOutput = tuple[np.ndarray, dict[str, Any]]


def uniform_quantize(data: np.ndarray, num_bits: int = 8) -> QuantizeOutput:
    """Quantize with uniform bins (implementation planned for Phase 2)."""
    raise NotImplementedError("uniform_quantize will be implemented in Phase 2.")


def minmax_quantize(data: np.ndarray, num_bits: int = 8) -> QuantizeOutput:
    """Quantize via min-max normalization (implementation planned for Phase 2)."""
    raise NotImplementedError("minmax_quantize will be implemented in Phase 2.")


def kmeans_quantize(data: np.ndarray, num_bits: int = 8) -> QuantizeOutput:
    """Quantize via centroid assignment (implementation planned for Phase 2)."""
    raise NotImplementedError("kmeans_quantize will be implemented in Phase 2.")


def percentile_quantize(data: np.ndarray, num_bits: int = 8) -> QuantizeOutput:
    """Quantize with percentile bins (implementation planned for Phase 2)."""
    raise NotImplementedError("percentile_quantize will be implemented in Phase 2.")


def dequantize(quantized: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Reconstruct float64 data from quantized representation (Phase 2)."""
    raise NotImplementedError("dequantize will be implemented in Phase 2.")
