"""Spectral Clustering model for the Smart Inference AI Fusion framework.

This module defines the SpectralClusteringModel class, a wrapper for scikit-learn's
SpectralClustering compatible with the BaseModel interface.
"""

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.logging import logger

# pylint: disable=duplicate-code


def _align_clusters_to_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Map predicted cluster ids to ground-truth labels using Hungarian assignment.

    Works with any hashable label dtype (ints/strings).

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted cluster ids.

    Returns:
        np.ndarray: y_pred remapped into y_true label space.
    """
    pred_vals = np.unique(y_pred)
    true_vals = np.unique(y_true)

    pred_index = {v: i for i, v in enumerate(pred_vals)}
    true_index = {v: i for i, v in enumerate(true_vals)}

    w = np.zeros((len(pred_vals), len(true_vals)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        w[pred_index[p], true_index[t]] += 1

    # maximize matches -> minimize (max - w)
    r, c = linear_sum_assignment(w.max() - w)
    mapping = {pred_vals[ri]: true_vals[ci] for ri, ci in zip(r, c)}

    return np.array([mapping[p] for p in y_pred], dtype=true_vals.dtype)


class SpectralClusteringModel(BaseModel):
    """Wrapper for scikit-learn's SpectralClustering, compatible with BaseModel."""

    def __init__(self, params: Optional[dict] = None, **kwargs: Any) -> None:
        """Initialize the SpectralClusteringModel."""
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = SpectralClustering(**params)
        self._fitted_data = None
        self._fitted_labels = None

    def train(self, X_train: ArrayLike, y_train: Optional[ArrayLike]) -> None:
        """Fit the Spectral Clustering model to the training data.

        Args:
            X_train (ArrayLike): Training features. Shape: (n_samples, n_features).
            y_train (ArrayLike | None): Ignored. Present for interface compatibility.
        """
        _ = y_train  # interface compatibility
        self._fitted_labels = self.model.fit_predict(X_train)
        self._fitted_data = np.asarray(X_train)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Fit the model to data (scikit-learn compatibility).

        Args:
            X (ArrayLike): The input data to fit the model on.
            y (Optional[ArrayLike], optional): Ignored. For API compatibility.
        """
        self.train(X, y)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for the input features.

        Since SpectralClustering doesn't have a native predict method,
        we use nearest neighbors to assign new points to existing clusters.

        Args:
            X (ArrayLike): The input data for which to predict clusters.

        Returns:
            np.ndarray: The predicted cluster labels.
        """
        if self._fitted_data is None or self._fitted_labels is None:
            raise ValueError("Model must be fitted before prediction")

        x_array = np.asarray(X)

        # If predicting on the same data used to fit, return cached labels
        if x_array.shape == self._fitted_data.shape and np.allclose(x_array, self._fitted_data):
            return self._fitted_labels.copy()

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self._fitted_data)
        _, indices = nn.kneighbors(x_array)
        return self._fitted_labels[indices.flatten()]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Estimate soft cluster assignment probabilities for each sample.

        Args:
            X (ArrayLike): The input data for which to predict probabilities.

        Returns:
            np.ndarray: The predicted cluster probabilities for each sample.
        """
        if self._fitted_data is None or self._fitted_labels is None:
            raise ValueError("Model must be fitted before prediction")

        x_array = np.asarray(X)
        n_clusters = len(np.unique(self._fitted_labels))

        # If it's the training set, return one-hot
        if x_array.shape == self._fitted_data.shape and np.allclose(x_array, self._fitted_data):
            probs = np.zeros((len(self._fitted_labels), n_clusters))
            probs[np.arange(len(self._fitted_labels)), self._fitted_labels] = 1.0
            return probs

        k = min(10, len(self._fitted_data))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self._fitted_data)
        distances, indices = nn.kneighbors(x_array)

        probs = np.zeros((len(x_array), n_clusters))
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            neighbor_labels = self._fitted_labels[idx]
            weights = 1.0 / (dist + 1e-8)  # avoid div/0
            for j in range(n_clusters):
                mask = neighbor_labels == j
                probs[i, j] = np.sum(weights[mask])
            s = probs[i].sum()
            probs[i] = probs[i] / s if s > 0 else np.full(n_clusters, 1.0 / n_clusters)
        return probs

    def evaluate(
        self,
        X_test: ArrayLike,
        y_test: Optional[ArrayLike] = None,
    ) -> Dict[str, Any]:
        """Evaluate the clustering model on the test set.

        Args:
            X_test (ArrayLike): The test features.
            y_test (Optional[ArrayLike]): The true labels for the test set.

        Returns:
            Dict[str, Any]: A dictionary of clustering and supervised metrics.
        """
        labels = self.predict(X_test)
        n_clusters = len(np.unique(self._fitted_labels)) if self._fitted_labels is not None else 0

        # Silhouette (safe)
        try:
            sil = silhouette_score(X_test, labels) if 1 < n_clusters < len(X_test) else None
        except ValueError:
            sil = None

        metrics: Dict[str, Any] = {
            "silhouette_score": sil,
            "n_clusters": float(n_clusters),
            "ari": None,
            "nmi": None,
            "accuracy": None,
            "balanced_accuracy": None,
            "f1": None,
            "accuracy_raw": None,
            "balanced_accuracy_raw": None,
            "f1_raw": None,
        }

        if y_test is not None:
            y_true = np.asarray(y_test)
            y_pred = np.asarray(labels)

            # Clustering metrics (label-invariant)
            try:
                metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
                metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
            except ValueError as exc:
                logger.warning("[SpectralClusteringModel] ARI/NMI failed: %s", exc)

            # Raw supervised metrics (no alignment)
            try:
                metrics["accuracy_raw"] = float(accuracy_score(y_true, y_pred))
                metrics["balanced_accuracy_raw"] = float(balanced_accuracy_score(y_true, y_pred))
                metrics["f1_raw"] = float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                )
            except ValueError:
                # keep as None if labels sets differ
                pass

            # Aligned supervised metrics (Hungarian)
            try:
                y_pred_aligned = _align_clusters_to_labels(y_true, y_pred)
                metrics["accuracy"] = float(accuracy_score(y_true, y_pred_aligned))
                metrics["balanced_accuracy"] = float(
                    balanced_accuracy_score(y_true, y_pred_aligned)
                )
                metrics["f1"] = float(
                    f1_score(y_true, y_pred_aligned, average="macro", zero_division=0)
                )
            except ValueError as exc:
                logger.warning("[SpectralClusteringModel] Supervised metrics failed: %s", exc)

        return metrics

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                         contained subobjects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params: Any) -> "SpectralClusteringModel":
        """Set the parameters of this estimator.

        Args:
            **params (Any): Estimator parameters.

        Returns:
            SpectralClusteringModel: The estimator instance.
        """
        self.model.set_params(**params)
        return self
