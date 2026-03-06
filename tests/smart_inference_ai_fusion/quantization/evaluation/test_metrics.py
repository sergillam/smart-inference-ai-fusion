"""Tests for quantization evaluation metrics."""

import numpy as np
import pytest

from smart_inference_ai_fusion.quantization.evaluation.metrics import (
    compute_clustering_metrics,
    compute_quantization_mse,
    compute_supervised_metrics,
)


def test_compute_supervised_metrics_perfect_predictions() -> None:
    """Degradation should be zero when baseline and quantized predictions match perfectly."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    result = compute_supervised_metrics(y_true, y_pred, y_pred)
    assert result["baseline_accuracy"] == 1.0
    assert result["quantized_accuracy"] == 1.0
    assert result["accuracy_degradation"] == 0.0


def test_compute_supervised_metrics_detects_degradation() -> None:
    """Accuracy degradation should be positive when quantized predictions worsen."""
    y_true = np.array([0, 1, 1, 0])
    y_baseline = np.array([0, 1, 1, 0])
    y_quantized = np.array([1, 1, 0, 0])
    result = compute_supervised_metrics(y_true, y_baseline, y_quantized)
    assert result["accuracy_degradation"] > 0.0
    assert result["f1_degradation"] >= 0.0


def test_compute_supervised_metrics_rejects_shape_mismatch() -> None:
    """Supervised metrics should validate input shapes."""
    with pytest.raises(ValueError, match="must match"):
        compute_supervised_metrics(np.array([0, 1]), np.array([0, 1]), np.array([0]))


def test_compute_supervised_metrics_rejects_empty_y_true() -> None:
    """Supervised metrics should reject empty targets."""
    with pytest.raises(ValueError, match="non-empty"):
        compute_supervised_metrics(np.array([]), np.array([]), np.array([]))


def test_compute_supervised_metrics_rejects_invalid_average() -> None:
    """Average mode must be one of macro/micro/weighted."""
    with pytest.raises(ValueError, match="average must be one of"):
        compute_supervised_metrics(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0, 1]),
            average="binary",  # type: ignore[arg-type]
        )


def test_compute_clustering_metrics_basic_keys_without_ground_truth() -> None:
    """Clustering metrics should return silhouette-only metrics without labels_true."""
    x = np.array([[0.0], [0.1], [1.0], [1.1]])
    labels_base = np.array([0, 0, 1, 1])
    labels_quant = np.array([0, 1, 1, 1])
    result = compute_clustering_metrics(x, labels_base, labels_quant)
    assert "baseline_silhouette" in result
    assert "quantized_silhouette" in result
    assert "baseline_ari" not in result


def test_compute_clustering_metrics_with_ground_truth_adds_ari_nmi() -> None:
    """When labels_true is provided, ARI/NMI metrics should be returned."""
    x = np.array([[0.0], [0.1], [1.0], [1.1]])
    labels_base = np.array([0, 0, 1, 1])
    labels_quant = np.array([0, 1, 1, 1])
    labels_true = np.array([0, 0, 1, 1])
    result = compute_clustering_metrics(x, labels_base, labels_quant, labels_true=labels_true)
    assert "baseline_ari" in result
    assert "quantized_ari" in result
    assert "baseline_nmi" in result
    assert "quantized_nmi" in result


def test_compute_clustering_metrics_returns_none_when_silhouette_invalid() -> None:
    """Silhouette should be None for single-cluster label assignments."""
    x = np.array([[0.0], [0.1], [1.0], [1.1]])
    labels_base = np.array([0, 0, 0, 0])
    labels_quant = np.array([0, 0, 0, 0])
    result = compute_clustering_metrics(x, labels_base, labels_quant)
    assert result["baseline_silhouette"] is None
    assert result["quantized_silhouette"] is None
    assert result["silhouette_degradation"] is None


def test_compute_clustering_metrics_rejects_label_shape_mismatch() -> None:
    """Clustering metrics should validate label shape compatibility."""
    x = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="must match"):
        compute_clustering_metrics(x, np.array([0, 1]), np.array([0]))


def test_compute_clustering_metrics_rejects_empty_x() -> None:
    """Clustering metrics should reject empty feature arrays."""
    x = np.empty((0, 2))
    with pytest.raises(ValueError, match="non-empty"):
        compute_clustering_metrics(x, np.array([]), np.array([]))


def test_compute_quantization_mse() -> None:
    """MSE helper should compute expected squared error."""
    original = np.array([0.0, 1.0, 2.0])
    reconstructed = np.array([0.0, 2.0, 2.0])
    mse = compute_quantization_mse(original, reconstructed)
    assert mse == pytest.approx(1.0 / 3.0)


def test_compute_quantization_mse_rejects_empty_input() -> None:
    """MSE helper should fail fast on empty inputs."""
    with pytest.raises(ValueError, match="original must be non-empty"):
        compute_quantization_mse(np.array([]), np.array([]))
