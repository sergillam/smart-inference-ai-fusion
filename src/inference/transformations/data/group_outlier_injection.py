import numpy as np
from sklearn.cluster import KMeans
from inference.transformations.data.base import InferenceTransformation

class GroupOutlierInjection(InferenceTransformation):
    """
    Injeta outliers em clusters/grupos específicos dos dados.
    Simula falhas estruturadas como desvios concentrados em regiões específicas.
    """

    def __init__(self, n_clusters=3, outlier_fraction=0.1, outlier_multiplier=10.0):
        self.n_clusters = n_clusters
        self.outlier_fraction = outlier_fraction
        self.outlier_multiplier = outlier_multiplier

    def apply(self, X):
        n_samples, n_features = X.shape
        n_outliers = int(n_samples * self.outlier_fraction)
        if n_outliers == 0:
            return X

        # Agrupa os dados em clusters
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        labels = kmeans.fit_predict(X)

        # Seleciona clusters aleatórios para injetar outliers
        selected_clusters = np.random.choice(self.n_clusters, size=1)
        indices = np.where(np.isin(labels, selected_clusters))[0]

        if len(indices) == 0:
            return X

        # Seleciona subconjunto dos índices para aplicar os outliers
        selected_indices = np.random.choice(indices, size=min(n_outliers, len(indices)), replace=False)

        # Injeta ruído forte nos pontos selecionados
        X_out = X.copy()
        noise = np.random.normal(loc=0.0, scale=self.outlier_multiplier, size=(len(selected_indices), n_features))
        X_out[selected_indices] += noise

        return X_out
