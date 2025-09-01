"""FastICA model for the Smart Inference AI Fusion framework.

This module defines the FastICAModel class, which combines scikit-learn's FastICA
for feature extraction with KMeans clustering.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.clustering import compute_clustering_metrics
from smart_inference_ai_fusion.utils.logging import logger


def _safe_asarray(arr: Any) -> np.ndarray:
    """Safely convert to numpy array, preserving original dtype when possible."""
    try:
        return np.asarray(arr)
    except (TypeError, ValueError):
        pass
    return arr  # Keep original dtype if conversion is not safe


def _coerce_labels_1d_int_safe(y: np.ndarray) -> np.ndarray:
    """Ensure the array is 1D and safely convert to int if possible without data loss."""
    arr = np.asarray(y).ravel()
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    try:
        as_int = arr.astype(int)
        # Round-trip to string to avoid converting non-integer strings
        if np.all(as_int.astype(str) == arr):
            return as_int
    except (TypeError, ValueError):
        pass
    return arr  # Keep original dtype if conversion is not safe


class FastICAModel(BaseModel):
    """A model combining FastICA for feature extraction and a KMeans head.

    This model is compatible with the BaseModel interface.

    Pipeline:
      - Train: Fit FastICA on X, transform X, then fit KMeans on the components.
      - Predict: Transform X using the fitted FastICA, then predict with KMeans.
    """

    def __init__(self, params: Optional[dict] = None, **kwargs: Any) -> None:
        """Initialize the FastICAModel.

        Args:
            params (dict | None): Parameters for the ``FastICAModel`` constructor.
                If ``None``, an empty dict is used.
            **kwargs: Additional keyword arguments for model parameters
                (merged into ``params``).
        """
        if params is None:
            params = {}
        params.update(kwargs)

        # Extract ICA parameters with safe defaults
        self._ica_params = {
            "n_components": params.get("n_components", None),
            "algorithm": params.get("algorithm", "parallel"),
            "whiten": params.get("whiten", "unit-variance"),
            "random_state": params.get("random_state", None),
            "max_iter": params.get("max_iter", 400),
            "tol": params.get("tol", 1e-4),
        }
        self._kmeans_params = {
            "n_clusters": params.get("kmeans_n_clusters", None),
            "n_init": int(params.get("kmeans_n_init", 10)),
            "random_state": params.get("kmeans_random_state", params.get("random_state", None)),
        }

        self.ica: Optional[FastICA] = FastICA(**self._ica_params)
        self.kmeans: Optional[KMeans] = None  # Created in train() once n_clusters is known

    def train(self, X_train: ArrayLike, y_train: Optional[ArrayLike] = None) -> None:
        """Fit FastICA on X_train and KMeans on the resulting components.

        Args:
            X_train (ArrayLike): The training features.
            y_train (Optional[ArrayLike]): Optional training labels, used to infer
                the number of clusters if not specified.
        """
        x_arr = np.asarray(X_train)

        # Decide number of components: if None, fallback to min(n_features, 64)
        if self.ica.n_components is None:
            n_features = x_arr.shape[1]
            self.ica.set_params(n_components=min(n_features, 64))

        # Fit FastICA and transform the data
        s_train = self.ica.fit_transform(x_arr)

        # Decide number of clusters for KMeans
        n_clusters = self._kmeans_params["n_clusters"]
        if n_clusters is None and y_train is not None:
            try:
                n_clusters = int(len(np.unique(np.asarray(y_train))))
            except (TypeError, ValueError):
                n_clusters = 10  # Fallback
        if n_clusters is None:
            n_clusters = 10

        self.kmeans = KMeans(
            n_clusters=int(max(2, n_clusters)),
            n_init=self._kmeans_params["n_init"],
            random_state=self._kmeans_params["random_state"],
        )
        self.kmeans.fit(s_train)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Fit the model to data (for scikit-learn compatibility).

        Args:
            X (ArrayLike): The input data to fit the model on.
            y (Optional[ArrayLike]): Optional labels.
        """
        self.train(X, y)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for the input features.

        Args:
            X (ArrayLike): The input data for which to predict clusters.

        Returns:
            np.ndarray: The predicted cluster labels.
        """
        if self.ica is None or self.kmeans is None:
            raise ValueError("Model must be fitted before prediction")
        x_arr = np.asarray(X)
        s = self.ica.transform(x_arr)
        return self.kmeans.predict(s)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return posterior probabilities of each mixture component for each sample.

        Args:
            X (ArrayLike): The input data for which to predict probabilities.

        Returns:
            np.ndarray: The predicted cluster probabilities for each sample.
        """
        if self.ica is None or self.kmeans is None:
            raise ValueError("Model must be fitted before prediction")
        x_arr = np.asarray(X)
        s = self.ica.transform(x_arr)
        distances = self.kmeans.transform(s)  # Shape: (n_samples, n_clusters)

        # Softmax over negative distance with a numerically stable fallback
        exp_neg = np.exp(-distances)
        denom = np.sum(exp_neg, axis=1, keepdims=True)
        probs = np.divide(
            exp_neg,
            denom,
            out=np.full_like(exp_neg, 1.0 / exp_neg.shape[1]),
            where=denom != 0,
        )
        return probs

    def _compute_supervised_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate all supervised metrics (raw, aligned, and clustering)."""
        metrics: Dict[str, Any] = {}

        # Label-invariant clustering metrics
        try:
            metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
            metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
        except ValueError as exc:
            logger.warning("[FastICAModel] ARI/NMI failed: %s", exc)

        # Raw supervised metrics (no alignment)
        try:
            metrics["accuracy_raw"] = float(accuracy_score(y_true, y_pred))
            metrics["balanced_accuracy_raw"] = float(balanced_accuracy_score(y_true, y_pred))
            metrics["f1_raw"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        except ValueError:
            # Keep as None if label spaces are incompatible
            metrics.setdefault("accuracy_raw", None)
            metrics.setdefault("balanced_accuracy_raw", None)
            metrics.setdefault("f1_raw", None)

        # Aligned supervised metrics using utility function
        try:
            supervised_metrics = compute_clustering_metrics(y_true, y_pred, use_hungarian=True)
            metrics.update(supervised_metrics)
        except ValueError as exc:
            logger.warning("[FastICAModel] Supervised metrics failed: %s", exc)

        return metrics

    def evaluate(self, X_test: ArrayLike, y_test: Optional[ArrayLike] = None) -> Dict[str, Any]:
        """Evaluate the model on test data using clustering and supervised metrics.

        Args:
            X_test (ArrayLike): The test features.
            y_test (Optional[ArrayLike]): The ground truth labels for the test set.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        if self.ica is None or self.kmeans is None:
            raise ValueError("Model must be fitted before evaluation")

        x_arr = np.asarray(X_test)
        s_test = self.ica.transform(x_arr)
        labels = self.kmeans.predict(s_test)

        # Silhouette score in the ICA component space
        try:
            sil = silhouette_score(s_test, labels) if s_test.shape[0] > 2 else None
        except ValueError:
            sil = None

        metrics: Dict[str, Any] = {
            "silhouette_score": sil,
            "n_components": (
                float(self.ica.n_components_)
                if hasattr(self.ica, "n_components_")
                else float(self.ica.n_components or 0)
            ),
        }

        # Supervised metrics (with safe label coercion)
        if y_test is not None:
            y_true = _coerce_labels_1d_int_safe(y_test)
            y_pred = _coerce_labels_1d_int_safe(labels)
            supervised_metrics = self._compute_supervised_metrics(y_true, y_pred)
            metrics.update(supervised_metrics)

        return metrics

    def get_params(self, _deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Returns:
            dict: Parameter names mapped to their values.
        """
        out = dict(self._ica_params)
        out.update(
            {
                "kmeans_n_clusters": self._kmeans_params["n_clusters"],
                "kmeans_n_init": self._kmeans_params["n_init"],
                "kmeans_random_state": self._kmeans_params["random_state"],
            }
        )
        return out

    def set_params(self, **params: Any) -> "FastICAModel":
        """Set the parameters of this estimator.

        Args:
            **params (Any): Estimator parameters.

        Returns:
            FastICAModel: The estimator instance.
        """
        if not params:
            return self
        # Set ICA params
        for k in ("n_components", "algorithm", "whiten", "random_state", "max_iter", "tol"):
            if k in params:
                self._ica_params[k] = params[k]
        self.ica.set_params(**self._ica_params)

        # Set KMeans params
        if "kmeans_n_clusters" in params:
            self._kmeans_params["n_clusters"] = params["kmeans_n_clusters"]
        if "kmeans_n_init" in params:
            self._kmeans_params["n_init"] = int(params["kmeans_n_init"])
        if "kmeans_random_state" in params or "random_state" in params:
            self._kmeans_params["random_state"] = params.get(
                "kmeans_random_state",
                params.get("random_state", self._kmeans_params["random_state"]),
            )
        return self
