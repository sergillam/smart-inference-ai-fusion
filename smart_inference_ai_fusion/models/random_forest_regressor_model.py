"""RandomForestRegressorModel for the Smart Inference AI Fusion framework.

This module defines the RandomForestRegressorModel class, a wrapper for
scikit-learn's RandomForestRegressor.
"""

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from smart_inference_ai_fusion.core.base_regression_model import BaseRegressionModel
from smart_inference_ai_fusion.utils import logging
from smart_inference_ai_fusion.utils.metrics import evaluate_regression


class RandomForestRegressorModel(BaseRegressionModel):
    """Random Forest Regressor model wrapper for the framework.

    This class wraps sklearn's RandomForestRegressor and exposes a consistent
    interface for regression tasks.
    """

    def __init__(self, params: Optional[dict] = None, **kwargs: Any) -> None:
        """Initialize the RandomForestRegressorModel.

        Args:
            params (dict | None): Parameters for the ``RandomForestRegressor`` constructor.
                If ``None``, an empty dict is used.
            **kwargs: Additional keyword arguments for model parameters (merged into ``params``).
        """
        super().__init__()
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = RandomForestRegressor(**params)

    def evaluate(self, X_test: Any, y_test: Any):
        """Evaluate the model on the test set with both regression and classification metrics.

        Args:
            X_test: Test features. Shape: ``(n_samples, n_features)``.
            y_test: True targets. Shape: ``(n_samples,)``.

        Returns:
            Dict[str, float]: Combined regression and classification metrics.
        """
        y_pred = self.model.predict(X_test)

        # Get regression metrics
        regression_metrics = evaluate_regression(y_test, y_pred)

        # Convert regression predictions to classification for additional metrics
        # Round predictions to nearest integer (suitable for digits dataset 0-9)
        y_pred_class = np.round(y_pred).astype(int)
        y_test_class = np.asarray(y_test, dtype=int)

        # Ensure predictions are within valid range [0, 9] for digits dataset
        y_pred_class = np.clip(y_pred_class, 0, 9)

        # Calculate classification metrics
        try:
            classification_metrics = {
                "accuracy": float(accuracy_score(y_test_class, y_pred_class)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test_class, y_pred_class)),
                "f1": float(f1_score(y_test_class, y_pred_class, average="macro", zero_division=0)),
                "precision": float(
                    precision_score(y_test_class, y_pred_class, average="macro", zero_division=0)
                ),
                "recall": float(
                    recall_score(y_test_class, y_pred_class, average="macro", zero_division=0)
                ),
            }

        except (ValueError, TypeError) as e:
            # Fallback to None if classification metrics fail
            classification_metrics = {
                "accuracy": None,
                "balanced_accuracy": None,
                "f1": None,
                "precision": None,
                "recall": None,
            }

            logging.warning("Classification metrics calculation failed: %s", e)

        # Combine both sets of metrics
        combined_metrics = {**regression_metrics, **classification_metrics}
        return combined_metrics

    def predict_proba(self, X):
        """Probability prediction is not supported for regressors.

        Args:
            X: Input features. Shape: ``(n_samples, n_features)``.

        Raises:
            AttributeError: Always, since regression models do not support ``predict_proba``.
        """
        raise AttributeError(
            "RandomForestRegressor doesn't support predict_proba for regression tasks."
        )
