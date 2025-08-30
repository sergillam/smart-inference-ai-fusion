"""MiniBatchKMeans model for the Smart Inference AI Fusion framework.

This module defines the MiniBatchKMeansModel class, a wrapper for scikit-learn's
MiniBatchKMeans compatible with the BaseModel interface.
"""

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    silhouette_score,
)

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.logging import logger

# pylint: disable=duplicate-code


class MiniBatchKMeansModel(BaseModel):
    """Wrapper for scikit-learn's MiniBatchKMeans, compatible with BaseModel.

    Allows use of evaluation and inference methods in the Smart Inference AI Fusion framework.
    """

    def __init__(self, params: Optional[dict] = None, **kwargs: Any) -> None:
        """Initialize the MiniBatchKMeansModel.

        Args:
            params (dict | None): Parameters for the ``MiniBatchKMeans`` constructor.
                If ``None``, an empty dict is used.
            **kwargs: Additional keyword arguments for model parameters
                (merged into ``params``).
        """
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = MiniBatchKMeans(**params)

    def train(self, X_train: ArrayLike, y_train: Optional[ArrayLike] = None) -> None:
        """Fit the Mini-Batch K-Means model to the training data.

        Args:
            X_train (ArrayLike): Training features.
                Shape: ``(n_samples, n_features)``.
            y_train (ArrayLike | None): Ignored. Present for interface compatibility.
        """
        _ = y_train  # interface compatibility
        self.model.fit(X_train)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Fit the model to data (scikit-learn compatibility).

        Args:
            X (ArrayLike): Feature matrix. Shape: ``(n_samples, n_features)``.
            y (ArrayLike | None): Ignored. Present for interface compatibility.
        """
        self.train(X, y)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for the input features.

        Args:
            X (ArrayLike): Features to predict.
                Shape: ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Predicted cluster labels. Shape: ``(n_samples,)``.
        """
        return self.model.predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Estimate soft cluster assignment probabilities for each sample.

        A softmax over negative distances is used as a heuristic probability.

        Args:
            X (ArrayLike): Input features.
                Shape: ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Probability distribution over clusters for each sample.
                Shape: ``(n_samples, n_clusters)``.
        """
        distances = self.model.transform(X)  # shape: (n_samples, n_clusters)
        exp_neg_dist = np.exp(-distances)
        row_sums = np.sum(exp_neg_dist, axis=1, keepdims=True)
        # Avoid divide-by-zero: fallback to uniform distribution where needed
        probs = np.divide(
            exp_neg_dist,
            row_sums,
            out=np.full_like(exp_neg_dist, 1.0 / exp_neg_dist.shape[1]),
            where=row_sums != 0,
        )
        return probs

    def evaluate(
        self,
        X_test: ArrayLike,
        y_test: Optional[ArrayLike] = None,
    ) -> Dict[str, Any]:
        """Evaluate the clustering model on the test set.

        Args:
            X_test (ArrayLike): Feature matrix for evaluation.
                Shape: ``(n_samples, n_features)``.
            y_test (ArrayLike | None): True labels (optional). If provided and
                comparable to predicted labels, supervised metrics are reported.

        Returns:
            Dict[str, Any]: Dictionary with clustering (and optional supervised) metrics:
                - ``inertia`` (float | None)
                - ``silhouette_score`` (float | None)
                - ``n_clusters`` (float)
                - ``accuracy`` (float | None)
                - ``balanced_accuracy`` (float | None)
                - ``f1`` (float | None)
        """
        labels = self.model.predict(X_test)
        inertia = self.model.inertia_

        # Silhouette only makes sense with >1 cluster and enough samples
        try:
            sil = (
                silhouette_score(X_test, labels)
                if self.model.n_clusters > 1 and len(X_test) > self.model.n_clusters
                else None
            )
        except ValueError:
            sil = None  # silhouette_score can raise if only one label is present

        metrics: Dict[str, Any] = {
            "inertia": float(inertia) if inertia is not None else None,
            "silhouette_score": sil,
            "n_clusters": float(self.model.n_clusters),
            "accuracy": None,
            "balanced_accuracy": None,
            "f1": None,
        }

        if y_test is not None:
            y_test_arr = np.asarray(y_test)
            labels_arr = np.asarray(labels)
            # Both must be comparable (same dtype kind) for supervised metrics
            if y_test_arr.dtype == labels_arr.dtype:
                try:
                    metrics["accuracy"] = float(accuracy_score(y_test, labels))
                    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, labels))
                    metrics["f1"] = float(f1_score(y_test, labels, average="macro"))
                except ValueError as exc:
                    # Keep supervised metrics as None if computation fails
                    logger.warning(
                        "[MiniBatchKMeansModel] Error computing supervised metrics: %s", exc
                    )
            else:
                logger.warning(
                    "[MiniBatchKMeansModel] Incompatible dtypes for y_test and labels. "
                    "Skipping supervised metrics."
                )

        return metrics

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Args:
            deep (bool): If ``True``, also return parameters of contained subobjects.

        Returns:
            dict: Model parameters.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params: Any) -> "MiniBatchKMeansModel":
        """Set the parameters of this estimator.

        Args:
            **params: Model parameters to set.

        Returns:
            MiniBatchKMeansModel: ``self`` to allow chaining.
        """
        self.model.set_params(**params)
        return self
