"""GaussianNB model wrapper with unified training and evaluation interface."""

from typing import Any, Dict, Literal

from sklearn.naive_bayes import GaussianNB

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.metrics import evaluate_classification


class GaussianNBModel(BaseModel):
    """Wrapper class for the Gaussian Naive Bayes model.

    Integrates with the framework's base model interface. Supports model initialization,
    training, and evaluation with standardized metrics and reporting.

    Attributes:
        model (GaussianNB): The underlying scikit-learn GaussianNB model.
    """

    def __init__(self, params: dict = None, **kwargs):
        """Initializes the GaussianNBModel.

        Args:
            params (dict, optional): Parameters for the GaussianNB constructor.
            **kwargs: Additional keyword arguments for model parameters.
        """
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = GaussianNB(**params)

    def train(self, X_train: Any, y_train: Any) -> None:
        """Fits the GaussianNB model to the training data.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
        """
        self.model.fit(X_train, y_train)

    def evaluate(
        self, X_test: Any, y_test: Any, average: Literal["macro", "micro", "weighted"] = "macro"
    ) -> Dict[str, Any]:
        """Evaluates the model on the test set.

        Args:
            X_test (Any):
                Input (features) data for testing.
            y_test (Any):
                True labels for testing.
            average (Literal["macro", "micro", "weighted"], optional):
                The averaging method for multiclass metrics.
                Default is "macro".

        Returns:
            Dict[str, Any]:
                A dictionary containing classification metrics such as
                accuracy, f1_score, precision, recall, etc.

        Example:
            >>> # Assuming 'model' is an instance of your class
            >>> # and X_test, y_test exist.
            >>> results = model.evaluate(X_test, y_test, average="macro")
            >>> print(results['accuracy'])
            0.93
        """
        predictions = self.model.predict(X_test)
        return evaluate_classification(y_test, predictions, average=average)
