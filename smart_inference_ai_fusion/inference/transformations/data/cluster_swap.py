"""Cluster-based feature perturbation transformation for structured data inference.

This module defines a transformation that swaps samples between clusters in the feature space,
introducing structured perturbations (not purely random noise), suitable for simulating
realistic anomalies in grouped or clustered datasets.
"""

import numpy as np
from sklearn.cluster import KMeans

from smart_inference_ai_fusion.inference.transformations.data.base import InferenceTransformation


class ClusterSwap(InferenceTransformation):
    """Swaps samples between clusters to simulate structured perturbations.

    This transformation perturbs the dataset by swapping a fraction of samples
    between discovered clusters, reflecting realistic structured anomalies in data.

    Attributes:
        n_clusters (int): The number of clusters to identify in the data.
        swap_fraction (float): The fraction of samples in each cluster to swap.
    """

    def __init__(self, n_clusters=3, swap_fraction=0.1):
        """Initializes the ClusterSwap transformation.

        Args:
            n_clusters (int, optional): Number of clusters to form. Defaults to 3.
            swap_fraction (float, optional): Fraction of samples in each cluster to swap.
                Defaults to 0.1.
        """
        self.n_clusters = n_clusters
        self.swap_fraction = swap_fraction

    def apply(self, X):
        """Applies the cluster swap perturbation to the input features.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Perturbed feature matrix with samples swapped across clusters.
        """
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

        if not to_swap:
            return X  # Nothing to swap

        # Flatten and shuffle
        flat_indices = np.concatenate(to_swap)
        shuffled = np.random.permutation(flat_indices)

        x_swapped = X.copy()
        x_swapped[flat_indices] = X[shuffled]
        return x_swapped
