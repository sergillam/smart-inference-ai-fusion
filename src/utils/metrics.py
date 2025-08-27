"""Evaluation metrics utilities for classification and clustering tasks."""

import numpy as np
from typing import Any, Dict, Optional, Literal
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    silhouette_score, mean_squared_error, mean_absolute_error, r2_score, median_absolute_error,
    explained_variance_score
)

def evaluate_regression(y_true, y_pred) -> dict:
    """Computes regression metrics for supervised regression models.

    Args:
        y_true (array-like): Ground truth (true) target values.
        y_pred (array-like): Predicted target values from the regressor.

    Returns:
        dict: A dictionary containing regression metrics:
            - mse: Mean Squared Error
            - mae: Mean Absolute Error
            - median_ae: Median Absolute Error
            - r2: R^2 (coefficient of determination)
            - explained_variance: Explained Variance Score
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
    y_true: Any,
    y_pred: Any,
    average: Literal["macro", "micro", "weighted"] = "macro"
) -> Dict[str, Any]:
    """Computes classification metrics for supervised models.

    This function handles pattern recognition, multiclass, or binary
    classification tasks.

    Args:
        y_true (Any):
            Ground truth (true) labels.
        y_pred (Any):
            Predicted labels from the classifier.
        average (Literal["macro", "micro", "weighted"], optional):
            Averaging method for multiclass metrics. Must be one of:
            "macro", "micro", or "weighted". Defaults to "macro".

    Returns:
        Dict[str, Any]: A dictionary containing the following metrics:
            - "accuracy": Overall accuracy.
            - "balanced_accuracy": Balanced accuracy.
            - "f1": F1 score (using the specified average).
            - "precision": Precision score (using the specified average).
            - "recall": Recall score (using the specified average).
            - "confusion_matrix": Confusion matrix as a nested list.

    Raises:
        ValueError: If `average` is not one of "macro", "micro", or "weighted".

    Example:
        >>> y_true = [0, 1, 2, 2]
        >>> y_pred = [0, 2, 1, 2]
        >>> evaluate_classification(y_true, y_pred, average="macro")
        {
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "f1": 0.4444444444444444,
            "precision": 0.5,
            "recall": 0.5,
            "confusion_matrix": [[1, 0, 0], [0, 0, 1], [0, 1, 1]]
        }
    """
    allowed_averages = ("macro", "micro", "weighted")
    if average not in allowed_averages:
        raise ValueError(
            f"Invalid average '{average}'. Must be one of {allowed_averages}."
        )

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
    y_true: Any,
    y_pred: Any,
    X: Optional[Any] = None
) -> Dict[str, Any]:
    """Computes common clustering (unsupervised) evaluation metrics.

    Args:
        y_true (array-like): Ground truth (class labels) or cluster assignments (if available).
        y_pred (array-like): Predicted cluster labels.
        X (array-like, optional): Feature data for silhouette score (optional).

    Returns:
        Dict[str, Any]: Dictionary with metrics: ARI, NMI, homogeneity, completeness, v_measure, 
            (optionally) silhouette.

    Example:
        >>> evaluate_clustering(y_true, y_pred, X)
        {'adjusted_rand_index': ..., ...}
    """
    metrics = {
        "adjusted_rand_index": adjusted_rand_score(y_true, y_pred),
        "normalized_mutual_info": normalized_mutual_info_score(y_true, y_pred),
        "homogeneity": homogeneity_score(y_true, y_pred),
        "completeness": completeness_score(y_true, y_pred),
        "v_measure": v_measure_score(y_true, y_pred),
    }
    if X is not None:
        metrics["silhouette"] = silhouette_score(X, y_pred)
    return metrics
