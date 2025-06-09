import numpy as np
from inference.transformations.label.base import LabelTransformation

class LabelFlipNearBorder(LabelTransformation):
    requires_model = True
    
    def __init__(self, flip_fraction):
        self.flip_fraction = flip_fraction
        self.requires_model = True

    def _get_confidence(self, model, X):
        """Recupera a confiança do modelo, tentando predict_proba, decision_function, ou modelo interno."""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            return np.max(probs, axis=1)
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X)
            return np.abs(decision)
        elif hasattr(model, "model"):  # suporta wrapper
            return self._get_confidence(model.model, X)
        else:
            raise ValueError("Model must support predict_proba or decision_function")

    def apply(self, y, X=None, model=None):
        if model is None or X is None:
            raise ValueError("This transformation requires model and X")

        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X has {len(X)} samples, y has {len(y)} labels")

        confidence = self._get_confidence(model, X)

        n = int(len(y) * self.flip_fraction)
        low_conf_indices = np.argsort(confidence)[:n]

        y_flipped = np.array(y).copy()
        classes = np.unique(y)
        for idx in low_conf_indices:
            available = classes[classes != y_flipped[idx]]
            y_flipped[idx] = np.random.choice(available)

        return y_flipped

