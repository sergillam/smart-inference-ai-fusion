from sklearn.svm import SVC
from utils.metrics import evaluate_all
from core.base_model import BaseModel

class SVMModel(BaseModel):
    def __init__(self, params=None, **kwargs):
        if params is None:
            params = {"kernel": "rbf", "C": 1.0}
        params.update(kwargs)
        self.model = SVC(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        return evaluate_all(y_test, predictions)
