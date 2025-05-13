# src/models/gaussian_model.py
from sklearn.naive_bayes import GaussianNB
from utils.metrics import evaluate_all
from core.base_model import BaseModel

class GaussianNBModel(BaseModel):
    def __init__(self, params=None, **kwargs):
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = GaussianNB(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        return evaluate_all(y_test, predictions)
