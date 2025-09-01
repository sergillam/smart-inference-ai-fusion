"""Base class for clustering models."""

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.clustering import (
    evaluate_clustering_model,
    predict_clustering_labels,
    predict_clustering_probabilities,
)
from smart_inference_ai_fusion.utils.model_utils import get_estimator_params, set_estimator_params


class BaseClusteringModel(BaseModel):
    """Base class for clustering models that provides common functionality."""

    def __init__(self) -> None:
        """Initialize the base clustering model."""
        super().__init__()
        self._fitted_data: Optional[np.ndarray] = None
        self._fitted_labels: Optional[np.ndarray] = None

    def train(self, X_train: ArrayLike, y_train: Optional[ArrayLike] = None) -> None:
        """Train the clustering model.

        Args:
            X_train (ArrayLike): Training features.
            y_train (Optional[ArrayLike]): Ignored for clustering.
        """
        _ = y_train  # interface compatibility
        self._fitted_labels = self.model.fit_predict(X_train)
        self._fitted_data = np.asarray(X_train)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Fit the model to data (scikit-learn compatibility).

        Args:
            X (ArrayLike): The input data to fit the model on.
            y (Optional[ArrayLike]): Ignored. For API compatibility.
        """
        self.train(X, y)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for the input features via nearest neighbors.

        Args:
            X (ArrayLike): The input data for which to predict clusters.

        Returns:
            np.ndarray: The predicted cluster labels.
        """
        if self._fitted_data is None or self._fitted_labels is None:
            raise ValueError("Model must be fitted before prediction")

        return predict_clustering_labels(self._fitted_data, self._fitted_labels, X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Estimate soft cluster assignment probabilities using k-NN distances.

        Args:
            X (ArrayLike): The input data for which to predict probabilities.

        Returns:
            np.ndarray: The predicted cluster probabilities for each sample.
                Shape: (n_samples, n_clusters).
        """
        if self._fitted_data is None or self._fitted_labels is None:
            raise ValueError("Model must be fitted before prediction")

        n_clusters = len(np.unique(self._fitted_labels))
        return predict_clustering_probabilities(
            self._fitted_data, self._fitted_labels, X, n_clusters
        )

    def evaluate(self, X_test: ArrayLike, y_test: Optional[ArrayLike] = None) -> Dict[str, Any]:
        """Evaluate the clustering model on the test set.

        Args:
            X_test (ArrayLike): The test features.
            y_test (Optional[ArrayLike]): The true labels for the test set.

        Returns:
            Dict[str, Any]: A dictionary of clustering and supervised metrics.
        """
        labels = self.predict(X_test)
        n_clusters = (
            int(len(np.unique(self._fitted_labels))) if self._fitted_labels is not None else 0
        )

        return evaluate_clustering_model(X_test, y_test, labels, n_clusters)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Args:
            deep (bool): If True, will return parameters for this estimator and subobjects.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return get_estimator_params(self.model, deep=deep)

    def set_params(self, **params: Any) -> "BaseClusteringModel":
        """Set the parameters of this estimator.

        Args:
            **params (Any): Estimator parameters.

        Returns:
            BaseClusteringModel: The estimator instance.
        """
        set_estimator_params(self.model, **params)
        return self
