"""Tests for feature-level dataset quantization."""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from smart_inference_ai_fusion.quantization.data.feature_quantizer import FeatureQuantizer


def test_transform_requires_fit() -> None:
    """Transform must fail when calibration was not performed."""
    quantizer = FeatureQuantizer(method="uniform", num_bits=8)
    with pytest.raises(RuntimeError, match="Call fit"):
        quantizer.transform(np.array([[1.0, 2.0]], dtype=np.float64))


def test_fit_transform_inverse_roundtrip_iris() -> None:
    """Roundtrip on Iris should keep low MSE for 16-bit uniform mode."""
    x, _ = load_iris(return_X_y=True)
    quantizer = FeatureQuantizer(method="uniform", num_bits=16)
    x_q = quantizer.fit_transform(x)
    x_r = quantizer.inverse_transform(x_q)
    mse = float(np.mean((x - x_r) ** 2))
    assert x_q.dtype == np.uint16
    assert mse < 0.01


def test_float16_path_casts_without_calibration() -> None:
    """float16 profile should perform cast-based quantization."""
    x = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    quantizer = FeatureQuantizer(method="uniform", num_bits=16, dtype_profile="float16")
    x_q = quantizer.fit_transform(x)
    x_r = quantizer.inverse_transform(x_q)
    assert x_q.dtype == np.float16
    assert x_r.dtype == np.float64


def test_float16_rejects_non_16bit_mode() -> None:
    """float16 profile only supports 16-bit mode."""
    with pytest.raises(ValueError, match="only supported with num_bits=16"):
        FeatureQuantizer(num_bits=8, dtype_profile="float16")


def test_transform_uses_fitted_parameters_without_recalibration() -> None:
    """Transform should use train-derived params, not test-set ranges."""
    x_train = np.array([[0.0], [1.0], [0.5]], dtype=np.float64)
    x_test = np.array([[-10.0], [2.0]], dtype=np.float64)

    quantizer = FeatureQuantizer(method="uniform", num_bits=8)
    quantizer.fit(x_train)
    params_before = dict(quantizer._params[0])  # pylint: disable=protected-access
    x_q = quantizer.transform(x_test)
    params_after = dict(quantizer._params[0])  # pylint: disable=protected-access

    assert params_before == params_after
    assert int(x_q.min()) == 0
    assert int(x_q.max()) == 255


def test_feature_count_mismatch_raises_clear_error() -> None:
    """Transform/inverse should reject arrays with different feature count."""
    x_train = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    quantizer = FeatureQuantizer(method="uniform", num_bits=8).fit(x_train)

    with pytest.raises(ValueError, match="Expected 2 features"):
        quantizer.transform(np.array([[0.5]], dtype=np.float64))

    with pytest.raises(ValueError, match="Expected 2 features"):
        quantizer.inverse_transform(np.array([[1]], dtype=np.uint8))
