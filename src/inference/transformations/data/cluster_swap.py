import numpy as np
from sklearn.cluster import KMeans
from inference.transformations.data.base import InferenceTransformation

class ClusterSwap(InferenceTransformation):
    """
    Swaps samples between clusters to simulate structured perturbations,
    which are not purely random and may reflect realistic anomalies in grouped data.
    """

    def __init__(self, n_clusters=3, swap_fraction=0.1):
        self.n_clusters = n_clusters
        self.swap_fraction = swap_fraction

    def apply(self, X):
        n_samples = len(X)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        labels = kmeans.fit_predict(X)
        clusters = [np.where(labels == i)[0] for i in range(self.n_clusters)]

        # Select a subset of samples to swap from each cluster
        to_swap = []
        for idxs in clusters:
            n_swap = int(len(idxs) * self.swap_fraction)
            if n_swap > 0:
                selected = np.random.choice(idxs, n_swap, replace=False)
                to_swap.append(selected)

        # Flatten and shuffle
        flat_indices = np.concatenate(to_swap)
        shuffled = np.random.permutation(flat_indices)

        X_swapped = X.copy()
        X_swapped[flat_indices] = X[shuffled]
        return X_swapped
