from sklearn.ensemble import GradientBoostingClassifier
from utils.metrics import evaluate_classification
from core.base_model import BaseModel

class GradientBoostingModel(BaseModel):
    """Gradient Boosting Classifier model wrapper for the Smart Inference AI Fusion framework.

    This class wraps sklearn's GradientBoostingClassifier and exposes a consistent interface.
    """
    def __init__(self, params: dict = None, **kwargs):
        """Initializes the GradientBoostingModel.

        Args:
            params (dict, optional): Parameters for the GradientBoostingClassifier constructor.
            **kwargs: Additional keyword arguments for model parameters.
        """
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = GradientBoostingClassifier(**params)

    def train(self, X_train, y_train):
        """Fits the Gradient Boosting model to the training data.

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
            average (str, optional): The averaging method for multiclass metrics. Default is "macro".

        Returns:
            dict: A dictionary containing classification metrics such as accuracy, f1_score, precision, recall, etc.
        """
        y_pred = self.model.predict(X_test)
        return evaluate_classification(y_test, y_pred, average=average)

    def fit(self, X, y):
        return self.train(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self
