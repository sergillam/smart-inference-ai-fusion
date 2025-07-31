"""GroupOutlierInjection transformation: injects outliers into specific clusters."""

import numpy as np
from sklearn.cluster import KMeans
from inference.transformations.data.base import InferenceTransformation

class GroupOutlierInjection(InferenceTransformation):
    """Injects outliers into specific clusters of the data.

    This transformation simulates structured failures, such as deviations
    concentrated in particular regions of the feature space.

    Attributes:
        n_clusters (int): Number of clusters used for grouping data.
        outlier_fraction (float): Fraction of total samples to inject as outliers.
        outlier_multiplier (float): Magnitude of outlier noise to be added.
    """

    def __init__(self, n_clusters=3, outlier_fraction=0.1, outlier_multiplier=10.0):
        """Initializes the GroupOutlierInjection transformation.

        Args:
            n_clusters (int, optional): Number of clusters to form. Default is 3.
            outlier_fraction (float, optional): Fraction of samples to turn into outliers.
                Default is 0.1.
            outlier_multiplier (float, optional): Standard deviation of the outlier noise.
                Default is 10.0.
        """
        self.n_clusters = n_clusters
        self.outlier_fraction = outlier_fraction
        self.outlier_multiplier = outlier_multiplier

    def apply(self, X):
        """Injects outliers into a selected cluster.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Feature matrix with injected outliers in specific clusters.
        """
        n_samples, n_features = X.shape
        n_outliers = int(n_samples * self.outlier_fraction)
        if n_outliers == 0:
            return X

        # Cluster the data
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        labels = kmeans.fit_predict(X)

        # Randomly select clusters for outlier injection
        selected_clusters = np.random.choice(self.n_clusters, size=1)
        indices = np.where(np.isin(labels, selected_clusters))[0]

        if len(indices) == 0:
            return X

        # Select a subset of samples from the cluster to receive outlier noise
        selected_indices = np.random.choice(
            indices, size=min(n_outliers, len(indices)), replace=False
        )

        # Inject strong noise into the selected points
        x_out = X.copy()
        noise = np.random.normal(
            loc=0.0,
            scale=self.outlier_multiplier,
            size=(len(selected_indices), n_features)
        )
        x_out[selected_indices] += noise

        return x_out
