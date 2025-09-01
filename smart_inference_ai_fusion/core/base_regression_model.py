"""Base regression model for the Smart Inference AI Fusion framework.

This module defines the BaseRegressionModel class, providing common functionality
for all regression models in the framework.
"""

from typing import Any, Dict

from numpy.typing import ArrayLike

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.metrics import evaluate_regression


class BaseRegressionModel(BaseModel):
    """Base class for all regression models."""

    def __init__(self) -> None:
        """Initialize the base regression model."""
        super().__init__()
        self.model = None

    def train(self, X_train: ArrayLike, y_train: ArrayLike) -> None:
        """Train the regression model.

        Args:
            X_train (ArrayLike): Training features.
                Shape: ``(n_samples, n_features)``.
            y_train (ArrayLike): Training targets.
                Shape: ``(n_samples,)``.
        """
        self.model.fit(X_train, y_train)

    def evaluate(
        self,
        X_test: ArrayLike,
        y_test: ArrayLike,
    ) -> Dict[str, Any]:
        """Evaluate the regression model on the test set.

        Args:
            X_test (ArrayLike): The feature matrix of the test set.
                Shape: ``(n_samples, n_features)``.
            y_test (ArrayLike): The true targets of the test set.
                Shape: ``(n_samples,)``.

        Returns:
            Dict[str, Any]: Dictionary with regression metrics.
        """
        y_pred = self.model.predict(X_test)
        return evaluate_regression(y_test, y_pred)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit the model to data (scikit-learn compatibility).

        Args:
            X (ArrayLike): Features. Shape: ``(n_samples, n_features)``.
            y (ArrayLike): Targets. Shape: ``(n_samples,)``.
        """
        self.train(X, y)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Generate predictions for the input features.

        Args:
            X (ArrayLike): Features to predict.
                Shape: ``(n_samples, n_features)``.

        Returns:
            ArrayLike: Predicted targets.
                Shape: ``(n_samples,)``.
        """
        return self.model.predict(X)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Args:
            deep (bool): If ``True``, return the parameters for this estimator
                and contained subobjects.

        Returns:
            dict: Model parameters.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params: Any) -> "BaseRegressionModel":
        """Set the parameters of this estimator.

        Args:
            **params: Model parameters to set.

        Returns:
            BaseRegressionModel: ``self`` to allow chaining.
        """
        self.model.set_params(**params)
        return self
