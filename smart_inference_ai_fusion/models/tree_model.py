"""Decision tree model wrapper with unified training and evaluation interface."""

from typing import Any, Dict, Literal

from sklearn.tree import DecisionTreeClassifier

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.metrics import evaluate_classification


class DecisionTreeModel(BaseModel):
    """Wrapper for the scikit-learn DecisionTreeClassifier.

    Provides a unified interface for training and evaluation using standard classification metrics.

    Attributes:
        model (DecisionTreeClassifier): The underlying scikit-learn decision tree model.
    """

    def __init__(self, params: dict = None, **kwargs):
        """Initializes the DecisionTreeModel.

        Args:
            params (dict, optional): Parameters for DecisionTreeClassifier.
                If None, uses {"max_depth": 1} by default.
            **kwargs: Additional keyword arguments for model parameters.
        """
        if params is None:
            params = {"max_depth": 1}
        params.update(kwargs)
        self.model = DecisionTreeClassifier(**params)

    def train(self, X_train: Any, y_train: Any) -> None:
        """Fits the DecisionTreeClassifier to the training data.

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
        """
        predictions = self.model.predict(X_test)
        return evaluate_classification(y_test, predictions, average=average)
