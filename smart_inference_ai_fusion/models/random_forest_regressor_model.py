"""RandomForestRegressorModel for the Smart Inference AI Fusion framework.

This module defines the RandomForestRegressorModel class, a wrapper for
scikit-learn's RandomForestRegressor.
"""

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestRegressor

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.metrics import evaluate_regression


# pylint: disable=duplicate-code
class RandomForestRegressorModel(BaseModel):
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
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = RandomForestRegressor(**params)

    def train(self, X_train: ArrayLike, y_train: ArrayLike) -> None:
        """Fit the Random Forest Regressor to the training data.

        Args:
            X_train (ArrayLike): Training features. Shape: ``(n_samples, n_features)``.
            y_train (ArrayLike): Training targets. Shape: ``(n_samples,)``.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: ArrayLike, y_test: ArrayLike) -> Dict[str, float]:
        """Evaluate the model on the test set.

        Args:
            X_test (ArrayLike): Test features. Shape: ``(n_samples, n_features)``.
            y_test (ArrayLike): True targets. Shape: ``(n_samples,)``.

        Returns:
            Dict[str, float]: Regression metrics, including:
                - ``mse``: Mean Squared Error
                - ``mae``: Mean Absolute Error
                - ``median_ae``: Median Absolute Error
                - ``r2``: Coefficient of determination
                - ``explained_variance``: Explained variance score
        """
        y_pred = self.model.predict(X_test)
        return evaluate_regression(y_test, y_pred)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:  # type: ignore[override]
        """Probability prediction is not supported for regressors.

        Args:
            X (ArrayLike): Input features. Shape: ``(n_samples, n_features)``.

        Raises:
            AttributeError: Always, since regression models do not support ``predict_proba``.
        """
        raise AttributeError(
            "RandomForestRegressor doesn't support predict_proba for regression tasks."
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit the model to data (scikit-learn compatibility).

        Args:
            X (ArrayLike): Feature matrix. Shape: ``(n_samples, n_features)``.
            y (ArrayLike): Target vector. Shape: ``(n_samples,)``.
        """
        self.train(X, y)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Generate predictions for the input features.

        Args:
            X (ArrayLike): Features to predict. Shape: ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Predicted targets. Shape: ``(n_samples,)``.
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

    def set_params(self, **params: Any) -> "RandomForestRegressorModel":
        """Set the parameters of this estimator.

        Args:
            **params: Model parameters to set.

        Returns:
            RandomForestRegressorModel: ``self`` to allow chaining.
        """
        self.model.set_params(**params)
        return self
