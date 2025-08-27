"""MLPModel module.

This module defines the MLPModel class, a wrapper for scikit-learn's
MLPClassifier to be used within the smart-inference-ai-fusion framework.
"""

from sklearn.neural_network import MLPClassifier
from utils.metrics import evaluate_classification
from core.base_model import BaseModel
# pylint: disable=duplicate-code
class MLPModel(BaseModel):
    """MLP Classifier model wrapper for the Smart Inference AI Fusion framework.

    This class wraps sklearn's MLPClassifier and exposes a consistent interface.
    """
    def __init__(self, params: dict = None, **kwargs):
        """Initializes the MLPModel.

        Args:
            params (dict, optional): Parameters for the MLPClassifier constructor.
            **kwargs: Additional keyword arguments for model parameters.
        """
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = MLPClassifier(**params)

    def train(self, X_train, y_train):
        """Fits the MLP model to the training data.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, average: str = "macro"):
        """Evaluates the model on the test set.

        Args:
            X_test (array-like): The feature matrix of the test set.
            y_test (array-like): The true labels of the test set.
            average (str, optional): The averaging method for multiclass metrics.
                Defaults to "macro".

        Returns:
            dict: A dictionary of classification metrics (e.g., accuracy, f1).
        """
        y_pred = self.model.predict(X_test)
        return evaluate_classification(y_test, y_pred, average=average)


    def fit(self, X, y):
        """Fit the model to data (scikit-learn compatibility).

        Args:
            X (array-like): Features.
            y (array-like): Labels.

        Returns:
            The result of the train method.
        """
        return self.train(X, y)

    def predict(self, X):
        """Generate predictions for the input features.

        Args:
            X (array-like): Features to predict.

        Returns:
            array-like: Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Args:
            X (array-like): Features to predict probabilities for.

        Returns:
            array-like: Predicted class probabilities.
        """
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator
            and contained subobjects.

        Returns:
            dict: Model parameters.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Args:
            **params: Model parameters to set.

        Returns:
            self
        """
        self.model.set_params(**params)
        return self
