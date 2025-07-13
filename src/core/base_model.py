"""Abstract base class for machine learning models in the framework."""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all machine learning models in the framework.

    Subclasses must implement the `train` and `evaluate` methods.
    Compatible with the scikit-learn interface for fitting and predicting.

    Attributes:
        model: The underlying estimator (set by subclasses, e.g., scikit-learn model).
    """

    model = None  # For pylint/type checkers (subclasses should overwrite)

    @abstractmethod
    def train(self, X_train, y_train):
        """Trains the model on the given data.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.

        Raises:
            NotImplementedError: If method not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement train method.")

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluates the model on the given test data.

        Args:
            X_test (array-like): Test features.
            y_test (array-like): Test labels.

        Returns:
            dict: Typically returns a dictionary with evaluation metrics.

        Raises:
            NotImplementedError: If method not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement evaluate method.")

    def fit(self, X, y):
        """Fits the model to data, for compatibility with scikit-learn.

        Args:
            X (array-like): Features.
            y (array-like): Labels.

        Returns:
            The result of the `train` method.
        """
        return self.train(X, y)

    def predict(self, X):
        """Generates predictions for the input features.

        Args:
            X (array-like): Features for prediction.

        Returns:
            array-like: Predictions from the underlying model.

        Raises:
            NotImplementedError: If the model does not implement `predict`.
        """
        if hasattr(self, "model") and hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise NotImplementedError("Model does not implement predict.")

    def predict_proba(self, X):
        """Generates class probabilities for the input features.

        Args:
            X (array-like): Features for prediction.

        Returns:
            array-like: Probabilities from the underlying model.

        Raises:
            AttributeError: If the model does not support `predict_proba`.
        """
        if hasattr(self, "model") and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("This model does not support predict_proba.")

    def decision_function(self, X):
        """Computes the distance of samples to the decision boundary.

        Args:
            X (array-like): Features for prediction.

        Returns:
            array-like: Decision function values from the underlying model.

        Raises:
            AttributeError: If the model does not support `decision_function`.
        """
        if hasattr(self, "model") and hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        raise AttributeError("This model does not support decision_function.")
