import numpy as np
from inference.transformations.label.base import LabelTransformation

class LabelFlipNearBorder(LabelTransformation):
    def __init__(self, model, X, fraction: float = 0.1):
        """
        Flip rótulos dos exemplos mais próximos à fronteira de decisão do modelo.

        Args:
            model: modelo treinado que implementa predict_proba() ou decision_function().
            X: dados de entrada para avaliar proximidade da borda.
            fraction: proporção de amostras a ter rótulo trocado.
        """
        self.model = model
        self.X = X
        self.fraction = fraction

    def apply(self, y):
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(self.X)
            confidence = np.max(probs, axis=1)
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(self.X)
            confidence = -np.abs(df)  # menos certeza = mais perto da borda
        else:
            raise ValueError("Model must support predict_proba or decision_function")

        y = np.array(y)
        n = int(len(y) * self.fraction)
        idx = np.argsort(confidence)[:n]  # menos confiante → mais próximo da borda
        classes = np.unique(y)

        y_flipped = y.copy()
        for i in idx:
            alt = classes[classes != y[i]]
            y_flipped[i] = np.random.choice(alt)

        return y_flipped
