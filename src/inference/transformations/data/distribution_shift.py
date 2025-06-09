import numpy as np
from inference.transformations.data.base import InferenceTransformation

class DistributionShiftMixing(InferenceTransformation):
    """
    Simula concept drift substituindo uma fração dos dados por pontos amostrados
    de uma distribuição diferente (ex: média deslocada).
    """

    def __init__(self, shift_fraction=0.1, shift_strength=2.0):
        self.shift_fraction = shift_fraction
        self.shift_strength = shift_strength

    def apply(self, X):
        n_samples = X.shape[0]
        n_shift = int(n_samples * self.shift_fraction)
        if n_shift == 0:
            return X

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        # Gera novos pontos deslocados da média
        shifted_samples = np.random.normal(loc=mean + self.shift_strength * std, scale=std, size=(n_shift, X.shape[1]))

        # Substitui aleatoriamente no conjunto original
        X_new = X.copy()
        indices = np.random.choice(n_samples, n_shift, replace=False)
        X_new[indices] = shifted_samples
        return X_new
