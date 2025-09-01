"""Utility functions for clustering models."""

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from smart_inference_ai_fusion.utils import logging


def align_clusters_to_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Map predicted cluster ids to ground-truth labels using Hungarian assignment.

    This function provides a robust mapping between cluster IDs and true labels,
    handling cases where there are more predicted clusters than true labels.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted cluster ids.

    Returns:
        np.ndarray: y_pred remapped into y_true label space.

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 2, 2])
        >>> y_pred = np.array([2, 2, 0, 0, 1, 1])  # Clusters in different order
        >>> aligned = align_clusters_to_labels(y_true, y_pred)
        >>> aligned
        array([0, 0, 1, 1, 2, 2])
    """
    pred_vals = np.unique(y_pred)
    true_vals = np.unique(y_true)

    pred_index = {v: i for i, v in enumerate(pred_vals)}
    true_index = {v: i for i, v in enumerate(true_vals)}

    # Create confusion matrix
    w = np.zeros((len(pred_vals), len(true_vals)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        w[pred_index[p], true_index[t]] += 1

    # Use Hungarian algorithm to find optimal assignment
    # We maximize matches, so minimize (max - w)
    row_idx, col_idx = linear_sum_assignment(w.max() - w)
    mapping = {pred_vals[ri]: true_vals[ci] for ri, ci in zip(row_idx, col_idx)}

    # Handle extra clusters that couldn't be mapped
    # Assign them to the most frequent true label
    if len(pred_vals) > len(true_vals):
        most_frequent_label = np.bincount(y_true).argmax()
        for pred_val in pred_vals:
            if pred_val not in mapping:
                mapping[pred_val] = most_frequent_label

    return np.array([mapping[p] for p in y_pred], dtype=true_vals.dtype)


def create_base_clustering_metrics(
    silhouette_score: Optional[float],
    n_clusters: int,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create base clustering metrics dictionary with safe defaults.

    Args:
        silhouette_score (Optional[float]): Computed silhouette score or None.
        n_clusters (int): Number of clusters.
        extra_metrics (Optional[Dict[str, Any]]): Additional metrics to include.

    Returns:
        Dict[str, Any]: Base metrics dictionary with default values.
    """
    metrics = {
        "silhouette_score": silhouette_score if silhouette_score is not None else 0.0,
        "n_clusters": float(n_clusters),
        "ari": 0.0,
        "nmi": 0.0,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "f1": 0.0,
        "accuracy_raw": 0.0,
        "balanced_accuracy_raw": 0.0,
        "f1_raw": 0.0,
    }

    if extra_metrics:
        metrics.update(extra_metrics)

    return metrics


def predict_clustering_labels(
    fitted_data: np.ndarray, fitted_labels: np.ndarray, X: ArrayLike
) -> np.ndarray:
    """Predict cluster labels for new data using nearest neighbors.

    Args:
        fitted_data (np.ndarray): Training data used to fit the model.
        fitted_labels (np.ndarray): Cluster labels from training.
        X (ArrayLike): New data to predict labels for.

    Returns:
        np.ndarray: Predicted cluster labels.
    """
    x_array = np.asarray(X)

    # If predicting on the same data used to fit, return cached labels
    if x_array.shape == fitted_data.shape and np.allclose(x_array, fitted_data):
        return fitted_labels.copy()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(fitted_data)
    _, indices = nn.kneighbors(x_array)
    return fitted_labels[indices.flatten()]


def predict_clustering_probabilities(
    fitted_data: np.ndarray, fitted_labels: np.ndarray, X: ArrayLike, n_clusters: int
) -> np.ndarray:
    """Predict cluster probabilities for new data using nearest neighbors.

    Args:
        fitted_data (np.ndarray): Training data used to fit the model.
        fitted_labels (np.ndarray): Cluster labels from training.
        X (ArrayLike): New data to predict probabilities for.
        n_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Predicted cluster probabilities.
    """
    x_array = np.asarray(X)

    # If predicting on the same data used to fit, return one-hot encoded labels
    if x_array.shape == fitted_data.shape and np.allclose(x_array, fitted_data):
        probs = np.zeros((len(x_array), n_clusters))
        for i, label in enumerate(fitted_labels):
            if 0 <= label < n_clusters:
                probs[i, label] = 1.0
        return probs

    k = min(10, len(fitted_data))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(fitted_data)
    distances, indices = nn.kneighbors(x_array)

    probs = np.zeros((len(x_array), n_clusters))
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        # Use inverse distance weighting
        weights = 1.0 / (dist + 1e-10)
        weights /= weights.sum()

        for j, neighbor_idx in enumerate(idx):
            neighbor_label = fitted_labels[neighbor_idx]
            if 0 <= neighbor_label < n_clusters:
                probs[i, neighbor_label] += weights[j]

    return probs


def compute_clustering_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, use_hungarian: bool = True
) -> dict:
    """Compute comprehensive clustering evaluation metrics.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted cluster labels.
        use_hungarian (bool): Whether to use Hungarian algorithm for alignment.

    Returns:
        dict: Dictionary containing various clustering metrics with default values.
    """
    # Initialize metrics with default values (0.0 instead of None for comparison)
    metrics = {
        "ari": 0.0,
        "nmi": 0.0,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "f1": 0.0,
        "accuracy_raw": 0.0,
        "balanced_accuracy_raw": 0.0,
        "f1_raw": 0.0,
    }

    # Compute unsupervised metrics
    try:
        metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
        metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
    except (ValueError, TypeError):
        # Keep default values (0.0) if computation fails
        pass

    # Compute supervised metrics with and without Hungarian alignment
    try:
        # Raw metrics (without alignment)
        metrics["accuracy_raw"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy_raw"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["f1_raw"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        if use_hungarian:
            # Aligned metrics (with Hungarian algorithm)
            y_pred_aligned = align_clusters_to_labels(y_true, y_pred)
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred_aligned))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred_aligned))
            metrics["f1"] = float(
                f1_score(y_true, y_pred_aligned, average="macro", zero_division=0)
            )
        else:
            metrics["accuracy"] = metrics["accuracy_raw"]
            metrics["balanced_accuracy"] = metrics["balanced_accuracy_raw"]
            metrics["f1"] = metrics["f1_raw"]

    except (ValueError, TypeError, IndexError) as e:
        # Keep default values (0.0) if metrics computation fails
        logging.warning("Supervised metrics computation failed: %s", e)
        # metrics already have default values, no need to update

    return metrics


def evaluate_clustering_model(
    X_test: ArrayLike,
    y_test: Optional[ArrayLike],
    labels: np.ndarray,
    n_clusters: int,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Common evaluation logic for clustering models.

    Args:
        X_test (ArrayLike): Test features.
        y_test (Optional[ArrayLike]): True labels (if available).
        labels (np.ndarray): Predicted cluster labels.
        n_clusters (int): Number of clusters.
        extra_metrics (Optional[Dict[str, Any]]): Additional model-specific metrics.

    Returns:
        Dict[str, Any]: Comprehensive clustering evaluation metrics.
    """
    # Silhouette score (safe computation)
    try:
        sil = silhouette_score(X_test, labels) if 1 < n_clusters < len(X_test) else None
    except ValueError:
        sil = None

    # Base metrics with safe defaults for comparison
    metrics = create_base_clustering_metrics(sil, n_clusters, extra_metrics)

    if y_test is not None:
        y_true = np.asarray(y_test)
        y_pred = np.asarray(labels)

        # Use utility function for all supervised metrics
        supervised_metrics = compute_clustering_metrics(y_true, y_pred, use_hungarian=True)
        metrics.update(supervised_metrics)

    return metrics
