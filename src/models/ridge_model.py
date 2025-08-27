"""RidgeModel module for the Smart Inference AI Fusion framework.

This module defines the RidgeModel class, a wrapper for scikit-learn's
RidgeClassifier to be used within the framework.
"""
from sklearn.linear_model import RidgeClassifier
import numpy as np
from scipy.special import softmax
from utils.metrics import evaluate_classification
from core.base_model import BaseModel
# pylint: disable=duplicate-code
class RidgeModel(BaseModel):
    """Ridge Classifier model wrapper for the Smart Inference AI Fusion framework.

    This class wraps sklearn's RidgeClassifier and exposes a consistent interface.
    """
    def __init__(self, params: dict = None, **kwargs):
        """Initializes the RidgeModel.

        Args:
            params (dict, optional): Parameters for the RidgeClassifier constructor.
            **kwargs: Additional keyword arguments for model parameters.
        """
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = RidgeClassifier(**params)

    def train(self, X_train, y_train):
        """Fits the Ridge model to the training data.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, average="macro"):
        """Evaluates the model on the test set.

        Args:
            X_test: Input (features) data for testing.
            y_test: True labels for testing.
            average (str, optional): The averaging method for multiclass metrics.
                Default is "macro".

        Returns:
            dict: A dictionary containing classification metrics such as accuracy,
            f1_score, precision, recall, etc.
        """
        y_pred = self.model.predict(X_test)
        return evaluate_classification(y_test, y_pred, average=average)

    def predict_proba(self, X):
        """Fallback probability prediction for RidgeClassifier.

        Uses the decision_function and applies softmax to approximate class
        probabilities. This enables compatibility with metrics and inference steps
        that require probabilities.

        Args:
            X (array-like): Input features.

        Returns:
            np.ndarray: Array of shape (n_samples, n_classes) with
            pseudo-probabilities.
        """
        if not hasattr(self.model, "decision_function"):
            raise AttributeError("RidgeClassifier does not support decision_function.")
        scores = self.model.decision_function(X)
        # For binary, decision_function returns shape (n_samples,); convert to (n_samples, 2)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        probs = softmax(scores, axis=1)
        return probs

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
