import numpy as np
from inference.transformations.data.base import InferenceTransformation

class TemporalDriftInjection(InferenceTransformation):
    """
    Simula um drift temporal aplicando ruído cumulativo em ordem crescente de amostras.
    Ideal para cenários com dados sequenciais ou sensoriais.
    """

    def __init__(self, drift_std=0.05):
        self.drift_std = drift_std

    def apply(self, X):
        n_samples, n_features = X.shape
        X_drifted = X.copy()

        # Drift progressivo com ruído crescente
        for i in range(n_samples):
            drift = np.random.normal(loc=0.0, scale=self.drift_std * (i / n_samples), size=n_features)
            X_drifted[i] += drift

        return X_drifted
