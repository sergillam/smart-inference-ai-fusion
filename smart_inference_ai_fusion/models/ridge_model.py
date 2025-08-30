"""RidgeModel module for the Smart Inference AI Fusion framework.

This module defines the RidgeModel class, a wrapper for scikit-learn's
RidgeClassifier to be used within the framework.
"""

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.metrics import evaluate_classification

# pylint: disable=duplicate-code


class RidgeModel(BaseModel):
    """Ridge Classifier model wrapper for the Smart Inference AI Fusion framework.

    This class wraps sklearn's ``RidgeClassifier`` and exposes a consistent interface.
    """

    def __init__(self, params: Optional[dict] = None, **kwargs: Any) -> None:
        """Initialize the RidgeModel.

        Args:
            params (dict | None): Parameters for the ``RidgeClassifier`` constructor.
                If ``None``, an empty dict is used.
            **kwargs: Additional keyword arguments merged into ``params``.
        """
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = RidgeClassifier(**params)

    def train(self, X_train: ArrayLike, y_train: ArrayLike) -> None:
        """Fit the Ridge model to the training data.

        Args:
            X_train (ArrayLike): Training features. Shape: ``(n_samples, n_features)``.
            y_train (ArrayLike): Training labels. Shape: ``(n_samples,)``.
        """
        self.model.fit(X_train, y_train)

    def evaluate(
        self, X_test: ArrayLike, y_test: ArrayLike, average: str = "macro"
    ) -> Dict[str, Any]:
        """Evaluate the model on the test set.

        Args:
            X_test (ArrayLike): Test features. Shape: ``(n_samples, n_features)``.
            y_test (ArrayLike): True labels. Shape: ``(n_samples,)``.
            average (str, optional): Averaging method for multiclass metrics.
                One of ``{"macro", "micro", "weighted"}``. Defaults to ``"macro"``.

        Returns:
            Dict[str, Any]: Classification metrics (accuracy, balanced_accuracy, f1, precision,
            recall, and confusion_matrix as nested list).
        """
        y_pred = self.model.predict(X_test)
        return evaluate_classification(y_test, y_pred, average=average)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Approximate class probabilities using ``decision_function`` + softmax.

        Args:
            X (ArrayLike): Input features. Shape: ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Pseudo-probabilities. Shape: ``(n_samples, n_classes)``.

        Raises:
            AttributeError: If the underlying estimator lacks ``decision_function``.
        """
        if not hasattr(self.model, "decision_function"):
            raise AttributeError("RidgeClassifier does not support decision_function.")
        scores = self.model.decision_function(X)
        # Binary case: shape (n_samples,) → make it (n_samples, 2)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        probs = softmax(scores, axis=1)
        return probs

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit the model to data (scikit-learn compatibility).

        Args:
            X (ArrayLike): Feature matrix. Shape: ``(n_samples, n_features)``.
            y (ArrayLike): Labels. Shape: ``(n_samples,)``.
        """
        self.train(X, y)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Generate predictions for the input features.

        Args:
            X (ArrayLike): Features to predict. Shape: ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Predicted labels. Shape: ``(n_samples,)``.
        """
        return self.model.predict(X)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Args:
            deep (bool): If ``True``, also return parameters of contained subobjects.

        Returns:
            dict: Model parameters.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params: Any) -> "RidgeModel":
        """Set the parameters of this estimator.

        Args:
            **params: Model parameters to set.

        Returns:
            RidgeModel: ``self`` to allow chaining.
        """
        self.model.set_params(**params)
        return self
