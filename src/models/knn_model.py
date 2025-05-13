from sklearn.neighbors import KNeighborsClassifier
from utils.metrics import evaluate_all
from core.base_model import BaseModel

class KNNModel(BaseModel):
    def __init__(self, params=None, **kwargs):
        if params is None:
            params = {"n_neighbors": 3}
        params.update(kwargs)
        self.model = KNeighborsClassifier(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        return evaluate_all(y_test, predictions)
