"""Base classification model for the Smart Inference AI Fusion framework.

This module defines the BaseClassificationModel class, providing common functionality
for all classification models in the framework.
"""

from typing import Any, Dict

from numpy.typing import ArrayLike

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.metrics import evaluate_classification


class BaseClassificationModel(BaseModel):
    """Base class for all classification models."""

    def __init__(self) -> None:
        """Initialize the base classification model."""
        super().__init__()
        self.model = None

    def train(self, X_train: ArrayLike, y_train: ArrayLike) -> None:
        """Train the classification model.

        Args:
            X_train (ArrayLike): Training features.
                Shape: ``(n_samples, n_features)``.
            y_train (ArrayLike): Training labels.
                Shape: ``(n_samples,)``.
        """
        self.model.fit(X_train, y_train)

    def evaluate(
        self,
        X_test: ArrayLike,
        y_test: ArrayLike,
        average: str = "macro",
    ) -> Dict[str, Any]:
        """Evaluate the classification model on the test set.

        Args:
            X_test (ArrayLike): The feature matrix of the test set.
                Shape: ``(n_samples, n_features)``.
            y_test (ArrayLike): The true labels of the test set.
                Shape: ``(n_samples,)``.
            average (str): Averaging method for multiclass metrics.
                Defaults to ``"macro"``.

        Returns:
            Dict[str, Any]: Dictionary with metrics
            (``accuracy``, ``balanced_accuracy``, ``f1``, ``precision``,
            ``recall``, ``confusion_matrix``).
        """
        y_pred = self.model.predict(X_test)
        return evaluate_classification(y_test, y_pred, average=average)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit the model to data (scikit-learn compatibility).

        Args:
            X (ArrayLike): Features. Shape: ``(n_samples, n_features)``.
            y (ArrayLike): Labels. Shape: ``(n_samples,)``.
        """
        self.train(X, y)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Generate predictions for the input features.

        Args:
            X (ArrayLike): Features to predict.
                Shape: ``(n_samples, n_features)``.

        Returns:
            ArrayLike: Predicted labels.
                Shape: ``(n_samples,)``.
        """
        return self.model.predict(X)

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Return class probability estimates for the input features.

        Args:
            X (ArrayLike): Features to predict probabilities for.
                Shape: ``(n_samples, n_features)``.

        Returns:
            ArrayLike: Probability estimates for each class.
                Shape: ``(n_samples, n_classes)``.
        """
        return self.model.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Args:
            deep (bool): If ``True``, return the parameters for this estimator
                and contained subobjects.

        Returns:
            dict: Model parameters.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params: Any) -> "BaseClassificationModel":
        """Set the parameters of this estimator.

        Args:
            **params: Model parameters to set.

        Returns:
            BaseClassificationModel: ``self`` to allow chaining.
        """
        self.model.set_params(**params)
        return self
