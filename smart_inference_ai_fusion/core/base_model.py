"""Abstract base class for machine learning models in the framework.

Defines the minimal interface each model must implement and provides thin
adapters to mimic parts of the scikit-learn API where appropriate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike


class BaseModel(ABC):
    """Abstract base class for all machine learning models in the framework.

    Subclasses must implement the :meth:`train` and :meth:`evaluate` methods.
    The class also offers convenience wrappers (``fit``, ``predict``,
    ``predict_proba``, ``decision_function``) to align with scikit-learn.

    Attributes:
        model (Any): Underlying estimator set by subclasses (e.g., a sklearn model).
    """

    model: Optional[Any] = None  # subclasses should overwrite after construction

    @abstractmethod
    def train(self, X_train: ArrayLike, y_train: ArrayLike) -> None:
        """Train the model on the given data.

        Args:
            X_train (ArrayLike): Training features.
                Shape: ``(n_samples, n_features)``.
            y_train (ArrayLike): Training labels/targets.
                Shape: ``(n_samples,)``.

        Returns:
            None: This method trains in-place and does not return a value.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement train().")

    @abstractmethod
    def evaluate(self, X_test: ArrayLike, y_test: ArrayLike) -> Dict[str, Any]:
        """Evaluate the model on the given test data.

        Args:
            X_test (ArrayLike): Test features.
                Shape: ``(n_samples, n_features)``.
            y_test (ArrayLike): Test labels/targets.
                Shape: ``(n_samples,)``.

        Returns:
            Dict[str, Any]: Dictionary with evaluation metrics. The exact keys depend
            on the task (e.g., accuracy/f1 for classification, mse/r2 for regression).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement evaluate().")

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit the model to data (scikit-learn-style alias to :meth:`train`).

        Args:
            X (ArrayLike): Feature matrix. Shape: ``(n_samples, n_features)``.
            y (ArrayLike): Labels/targets. Shape: ``(n_samples,)``.

        Returns: None
        """
        self.train(X, y)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Generate predictions for the input features.

        Args:
            X (ArrayLike): Features to predict.
                Shape: ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Predicted labels/targets. Shape: ``(n_samples,)``.

        Raises:
            AttributeError: If the underlying model does not implement ``predict``.
        """
        if hasattr(self, "model") and hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(X))
        raise AttributeError("Underlying model does not implement predict().")

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Generate class probabilities for the input features (if supported).

        Args:
            X (ArrayLike): Features to predict probabilities for.
                Shape: ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Class probabilities. Shape: ``(n_samples, n_classes)``.

        Raises:
            AttributeError: If the underlying model does not implement ``predict_proba``.
        """
        if hasattr(self, "model") and hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))
        raise AttributeError("This model does not support predict_proba().")

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Compute distance/scores to the decision boundary (if supported).

        Args:
            X (ArrayLike): Input features.
                Shape: ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Decision scores for each sample. Shape depends on estimator.

        Raises:
            AttributeError: If the underlying model does not implement ``decision_function``.
        """
        if hasattr(self, "model") and hasattr(self.model, "decision_function"):
            return np.asarray(self.model.decision_function(X))
        raise AttributeError("This model does not support decision_function().")
