"""Interface tests for quantization methods module."""

import numpy as np
import pytest

from smart_inference_ai_fusion.quantization.core import methods


@pytest.mark.parametrize(
    "fn_name",
    [
        "uniform_quantize",
        "minmax_quantize",
        "kmeans_quantize",
        "percentile_quantize",
    ],
)
def test_quantize_functions_raise_not_implemented(fn_name: str) -> None:
    """Phase 2 methods should exist and be explicitly unimplemented for now."""
    fn = getattr(methods, fn_name)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        fn(np.array([0.1, 0.2], dtype=np.float64), num_bits=8)


def test_dequantize_raises_not_implemented() -> None:
    """Dequantize interface should be present before implementation."""
    with pytest.raises(NotImplementedError, match="Phase 2"):
        methods.dequantize(np.array([1, 2], dtype=np.uint8), {"method": "uniform"})
