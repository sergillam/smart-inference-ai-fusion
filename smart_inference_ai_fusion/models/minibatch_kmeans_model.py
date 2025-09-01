"""MiniBatchKMeans model for the Smart Inference AI Fusion framework.

This module defines the MiniBatchKMeansModel class, a wrapper for scikit-learn's
MiniBatchKMeans compatible with the BaseModel interface.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import MiniBatchKMeans

from smart_inference_ai_fusion.core.base_clustering_model import BaseClusteringModel
from smart_inference_ai_fusion.utils.model_utils import get_estimator_params, set_estimator_params


class MiniBatchKMeansModel(BaseClusteringModel):
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
        super().__init__()

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

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Args:
            deep (bool): If ``True``, also return parameters of contained subobjects.

        Returns:
            dict: Model parameters.
        """
        return get_estimator_params(self.model, deep=deep)

    def set_params(self, **params: Any) -> "MiniBatchKMeansModel":
        """Set the parameters of this estimator.

        Args:
            **params: Model parameters to set.

        Returns:
            MiniBatchKMeansModel: ``self`` to allow chaining.
        """
        set_estimator_params(self.model, **params)
        return self
