"""Logistic Regression model wrapper with unified training and evaluation interface."""

from typing import Any, Dict, Literal

from sklearn.linear_model import LogisticRegression

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.metrics import evaluate_classification


class LogisticRegressionModel(BaseModel):
    """Wrapper for the scikit-learn LogisticRegression classifier.

    Provides a unified interface for training and evaluation using standard classification metrics.

    Attributes:
        model (LogisticRegression): The underlying scikit-learn logistic regression model.
    """

    def __init__(self, params: dict = None, **kwargs):
        """Initializes the LogisticRegressionModel.

        Args:
            params (dict, optional): Parameters for LogisticRegression.
                If None, uses default parameters optimized for stability.
            **kwargs: Additional keyword arguments for model parameters.
        """
        if params is None:
            params = {
                "random_state": 42,
                "max_iter": 1000,  # Increased for convergence
                "solver": "lbfgs",  # Stable solver for small datasets
            }
        params.update(kwargs)
        self.model = LogisticRegression(**params)

    def train(self, X_train: Any, y_train: Any) -> None:
        """Fits the LogisticRegression to the training data.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
        """
        self.model.fit(X_train, y_train)

    def evaluate(
        self, X_test: Any, y_test: Any, average: Literal["macro", "micro", "weighted"] = "macro"
    ) -> Dict[str, Any]:
        """Evaluates the LogisticRegression on test data.

        Args:
            X_test (array-like): Test features.
            y_test (array-like): Test labels.
            average: Averaging strategy for multi-class classification metrics.

        Returns:
            Dict[str, Any]: Dictionary containing classification metrics.
        """
        predictions = self.model.predict(X_test)
        return evaluate_classification(y_test, predictions, average=average)
