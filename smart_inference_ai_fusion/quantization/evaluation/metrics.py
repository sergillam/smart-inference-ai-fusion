"""Evaluation metrics for SIP-Q quantization experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)


def compute_supervised_metrics(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_quantized: np.ndarray,
    average: Literal["macro", "micro", "weighted"] = "weighted",
) -> dict[str, float]:
    """Compute baseline/quantized classification metrics and degradations."""
    y_true_arr = np.asarray(y_true)
    y_base = np.asarray(y_pred_baseline)
    y_quant = np.asarray(y_pred_quantized)
    _validate_average(average)
    if y_true_arr.size == 0:
        raise ValueError("y_true must be non-empty.")
    _validate_same_shape(y_true_arr, y_base, "y_true", "y_pred_baseline")
    _validate_same_shape(y_true_arr, y_quant, "y_true", "y_pred_quantized")

    acc_base = accuracy_score(y_true_arr, y_base)
    acc_quant = accuracy_score(y_true_arr, y_quant)
    f1_base = f1_score(y_true_arr, y_base, average=average, zero_division=0)
    f1_quant = f1_score(y_true_arr, y_quant, average=average, zero_division=0)
    prec_base = precision_score(y_true_arr, y_base, average=average, zero_division=0)
    prec_quant = precision_score(y_true_arr, y_quant, average=average, zero_division=0)
    rec_base = recall_score(y_true_arr, y_base, average=average, zero_division=0)
    rec_quant = recall_score(y_true_arr, y_quant, average=average, zero_division=0)

    return {
        "baseline_accuracy": float(acc_base),
        "quantized_accuracy": float(acc_quant),
        "accuracy_degradation": float(acc_base - acc_quant),
        "baseline_f1": float(f1_base),
        "quantized_f1": float(f1_quant),
        "f1_degradation": float(f1_base - f1_quant),
        "baseline_precision": float(prec_base),
        "quantized_precision": float(prec_quant),
        "precision_degradation": float(prec_base - prec_quant),
        "baseline_recall": float(rec_base),
        "quantized_recall": float(rec_quant),
        "recall_degradation": float(rec_base - rec_quant),
    }


def compute_clustering_metrics(
    x: np.ndarray,
    labels_baseline: np.ndarray,
    labels_quantized: np.ndarray,
    labels_true: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute clustering metrics and degradation for baseline/quantized labels."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_base = np.asarray(labels_baseline)
    y_quant = np.asarray(labels_quantized)
    if x_arr.size == 0:
        raise ValueError("x must be non-empty.")
    _validate_same_shape(y_base, y_quant, "labels_baseline", "labels_quantized")

    sil_base = _safe_silhouette(x_arr, y_base)
    sil_quant = _safe_silhouette(x_arr, y_quant)
    result: dict[str, float | None] = {
        "baseline_silhouette": sil_base,
        "quantized_silhouette": sil_quant,
        "silhouette_degradation": _safe_subtract(sil_base, sil_quant),
    }

    if labels_true is not None:
        y_true = np.asarray(labels_true)
        _validate_same_shape(y_true, y_base, "labels_true", "labels_baseline")
        ari_base = adjusted_rand_score(y_true, y_base)
        ari_quant = adjusted_rand_score(y_true, y_quant)
        nmi_base = normalized_mutual_info_score(y_true, y_base)
        nmi_quant = normalized_mutual_info_score(y_true, y_quant)
        result.update(
            {
                "baseline_ari": float(ari_base),
                "quantized_ari": float(ari_quant),
                "ari_degradation": float(ari_base - ari_quant),
                "baseline_nmi": float(nmi_base),
                "quantized_nmi": float(nmi_quant),
                "nmi_degradation": float(nmi_base - nmi_quant),
            }
        )

    return result


def compute_quantization_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute mean squared error between original and reconstructed tensors."""
    original_arr = np.asarray(original, dtype=np.float64)
    reconstructed_arr = np.asarray(reconstructed, dtype=np.float64)
    _validate_same_shape(original_arr, reconstructed_arr, "original", "reconstructed")
    return float(np.mean((original_arr - reconstructed_arr) ** 2))


def _validate_same_shape(
    left: np.ndarray, right: np.ndarray, left_name: str, right_name: str
) -> None:
    if left.shape != right.shape:
        raise ValueError(
            f"{left_name}.shape {left.shape} must match {right_name}.shape {right.shape}."
        )


def _safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float | None:
    if x.ndim != 2:
        raise ValueError("x must be a 2D array for silhouette computation.")
    if len(x) != len(labels):
        raise ValueError("x and labels must have the same number of samples.")

    unique = np.unique(labels)
    if len(unique) <= 1 or len(unique) >= len(labels):
        return None
    return float(silhouette_score(x, labels))


def _safe_subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _validate_average(average: str) -> None:
    valid = {"macro", "micro", "weighted"}
    if average not in valid:
        raise ValueError(f"average must be one of {valid}.")
