"""Tests for quantization core type aliases."""

from typing import get_args

from smart_inference_ai_fusion.quantization.core.types import BitWidth, DTypeProfile, QuantMethod


def test_quant_method_values() -> None:
    """Quantization methods should match SIP-Q plan."""
    assert get_args(QuantMethod) == ("uniform", "minmax", "kmeans", "percentile")


def test_bit_width_values() -> None:
    """Bit widths should support the core triplet 8/16/32."""
    assert get_args(BitWidth) == (8, 16, 32)


def test_dtype_profile_values() -> None:
    """DType profile should expose integer and float16 tracks."""
    assert get_args(DTypeProfile) == ("integer", "float16")
