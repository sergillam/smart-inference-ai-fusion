"""RandomForestRegressorModel module for the Smart Inference AI Fusion framework.

This module defines the RandomForestRegressorModel class, a wrapper for
scikit-learn's RandomForestRegressor.
"""
from sklearn.ensemble import RandomForestRegressor
from utils.metrics import evaluate_regression
from core.base_model import BaseModel
# pylint: disable=duplicate-code
class RandomForestRegressorModel(BaseModel):
    """Random Forest Regressor model wrapper for the framework.

    This class wraps sklearn's RandomForestRegressor and exposes a consistent
    interface for regression tasks.
    """
    def __init__(self, params: dict = None, **kwargs):
        """Initializes the RandomForestRegressorModel.

        Args:
            params (dict, optional): Parameters for the RandomForestRegressor constructor.
            **kwargs: Additional keyword arguments for model parameters.
        """
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = RandomForestRegressor(**params)

    def train(self, X_train, y_train):
        """Fits the Random Forest Regressor model to the training data.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training targets.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluates the model on the test set.

        Args:
            X_test (array-like): Input (features) data for testing.
            y_test (array-like): True targets for testing.

        Returns:
            dict: A dictionary containing regression metrics.
        """
        y_pred = self.model.predict(X_test)
        return evaluate_regression(y_test, y_pred)
    def predict_proba(self, X):
        """Probability prediction is not supported for regressors.

        Args:
            X (array-like): Input features.

        Raises:
            AttributeError: Always, since regression models do not support predict_proba.
        """
        raise AttributeError("RandomForestRegressor doesn't support predict_proba regression task.")


    def fit(self, X, y):
        """Fits the model to data (scikit-learn compatibility)."""
        return self.train(X, y)

    def predict(self, X):
        """Generates predictions for the input features."""
        return self.model.predict(X)

    def get_params(self, deep=True):
        """Gets the parameters for this estimator."""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Sets the parameters of this estimator."""
        self.model.set_params(**params)
        return self
