"""Utility functions for clustering model evaluation."""

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from smart_inference_ai_fusion.utils.logging import logger


def map_clusters_to_labels(cluster_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """Map cluster labels to true labels using Hungarian algorithm.

    This finds the optimal assignment of cluster IDs to true class labels
    that maximizes accuracy.

    Args:
        cluster_labels: Predicted cluster assignments
        true_labels: Ground truth class labels

    Returns:
        Mapped cluster labels that align with true labels
    """
    # Get unique labels
    unique_clusters = np.unique(cluster_labels)
    unique_true = np.unique(true_labels)

    # Create confusion matrix
    n_clusters = len(unique_clusters)
    n_classes = len(unique_true)

    # Build cost matrix (negative of counts for minimization)
    cost_matrix = np.zeros((n_clusters, n_classes))

    for i, cluster in enumerate(unique_clusters):
        for j, true_class in enumerate(unique_true):
            # Count how many samples with this cluster belong to this true class
            mask = cluster_labels == cluster
            count = np.sum(true_labels[mask] == true_class)
            cost_matrix[i, j] = -count  # Negative for minimization

    # Find optimal assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Create mapping dictionary
    cluster_to_label = {}
    for i, j in zip(row_indices, col_indices):
        cluster_to_label[unique_clusters[i]] = unique_true[j]

    # Map cluster labels to true labels
    mapped_labels = np.array([cluster_to_label.get(label, label) for label in cluster_labels])

    return mapped_labels


def calculate_supervised_clustering_metrics(
    predicted_labels: ArrayLike,
    true_labels: Optional[ArrayLike],
    model_name: str = "ClusteringModel",
) -> Dict[str, Any]:
    """Calculate supervised metrics for clustering models.

    Args:
        predicted_labels: Cluster assignments from the model
        true_labels: Ground truth class labels (optional)
        model_name: Name of the model for logging

    Returns:
        Dictionary with accuracy, balanced_accuracy, and f1 metrics
    """
    metrics = {
        "accuracy": None,
        "balanced_accuracy": None,
        "f1": None,
    }

    if true_labels is not None:
        try:
            y_true = np.asarray(true_labels)
            y_pred = np.asarray(predicted_labels)

            # Map clusters to true labels using Hungarian algorithm
            mapped_labels = map_clusters_to_labels(y_pred, y_true)

            # Calculate supervised metrics with mapped labels
            metrics["accuracy"] = float(accuracy_score(y_true, mapped_labels))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, mapped_labels))
            metrics["f1"] = float(f1_score(y_true, mapped_labels, average="macro"))

            logger.info(
                "[%s] Calculated supervised metrics using optimal cluster-to-label mapping",
                model_name,
            )

        except (ValueError, TypeError, IndexError) as exc:
            logger.warning("[%s] Error computing supervised metrics: %s", model_name, exc)

    return metrics
