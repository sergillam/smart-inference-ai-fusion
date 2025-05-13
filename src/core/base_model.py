from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """Treina o modelo com os dados de treinamento"""
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo usando o conjunto de teste.

        Retorna:
            dict: dicionário com métricas de desempenho, como:
            {
                'accuracy': 0.95,
                'f1': 0.94,
                ...
            }
        """
        pass
