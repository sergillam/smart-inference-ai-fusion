from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass

    def fit(self, X, y):
        """Compatibilidade com scikit-learn"""
        return self.train(X, y)

    def predict(self, X):
        if hasattr(self, "model") and hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise NotImplementedError("Model does not implement predict")

    def predict_proba(self, X):
        if hasattr(self, "model") and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("This model does not support predict_proba")

    def decision_function(self, X):
        if hasattr(self, "model") and hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        raise AttributeError("This model does not support decision_function")
