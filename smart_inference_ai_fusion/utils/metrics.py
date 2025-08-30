"""Evaluation metrics utilities for classification and clustering tasks."""

from typing import Any, Dict, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    completeness_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    homogeneity_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    normalized_mutual_info_score,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
    v_measure_score,
)


def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """Compute regression metrics for supervised regression models.

    Args:
        y_true (ArrayLike): Ground-truth (true) target values.
        y_pred (ArrayLike): Predicted target values from the regressor.

    Returns:
        Dict[str, float]: A dictionary containing:
            - "mse": Mean Squared Error.
            - "mae": Mean Absolute Error.
            - "median_ae": Median Absolute Error.
            - "r2": R-squared (coefficient of determination).
            - "explained_variance": Explained Variance Score.
    """
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
    }


def evaluate_classification(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: Literal["macro", "micro", "weighted"] = "macro",
) -> Dict[str, Any]:
    """Compute classification metrics for supervised models.

    Handles binary, multiclass e pattern recognition tasks.

    Args:
        y_true (ArrayLike): Ground-truth labels.
        y_pred (ArrayLike): Predicted labels from the classifier.
        average (Literal["macro", "micro", "weighted"], optional):
            Averaging method for multiclass metrics. Must be one of
            "macro", "micro", or "weighted". Defaults to "macro".

    Returns:
        Dict[str, Any]: A dictionary with:
            - "accuracy": Overall accuracy.
            - "balanced_accuracy": Balanced accuracy.
            - "f1": F1 score (using the specified average).
            - "precision": Precision score (using the specified average).
            - "recall": Recall score (using the specified average).
            - "confusion_matrix": Confusion matrix as a nested list.

    Raises:
        ValueError: If ``average`` is not one of "macro", "micro", or "weighted".
    """
    allowed_averages = ("macro", "micro", "weighted")
    if average not in allowed_averages:
        raise ValueError(f"Invalid average '{average}'. Must be one of {allowed_averages}.")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def evaluate_clustering(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    X: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    """Compute common clustering (unsupervised) evaluation metrics.

    Args:
        y_true (ArrayLike): Ground-truth class labels or known cluster assignments (if available).
        y_pred (ArrayLike): Predicted cluster labels.
        X (Optional[ArrayLike], optional): Feature matrix used to compute silhouette score.
            Shape: (n_samples, n_features). If not provided, silhouette is omitted.

    Returns:
        Dict[str, Any]: Dictionary with metrics:
            - "adjusted_rand_index"
            - "normalized_mutual_info"
            - "homogeneity"
            - "completeness"
            - "v_measure"
            - "silhouette" (only if ``X`` is provided)
    """
    metrics: Dict[str, Any] = {
        "adjusted_rand_index": adjusted_rand_score(y_true, y_pred),
        "normalized_mutual_info": normalized_mutual_info_score(y_true, y_pred),
        "homogeneity": homogeneity_score(y_true, y_pred),
        "completeness": completeness_score(y_true, y_pred),
        "v_measure": v_measure_score(y_true, y_pred),
    }
    if X is not None:
        metrics["silhouette"] = silhouette_score(X, y_pred)
    return metrics
