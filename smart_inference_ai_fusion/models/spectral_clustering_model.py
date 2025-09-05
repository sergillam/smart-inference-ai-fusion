"""Spectral Clustering model for the Smart Inference AI Fusion framework.

This module defines the SpectralClusteringModel class, a wrapper for scikit-learn's
SpectralClustering compatible with the BaseModel interface.
"""

from typing import Any, Optional

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import MinMaxScaler

from smart_inference_ai_fusion.core.base_clustering_model import BaseClusteringModel
from smart_inference_ai_fusion.utils.logging import logger


class SpectralClusteringModel(BaseClusteringModel):
    """Spectral Clustering model with robustness checks.

    Wrapper around scikit-learn's SpectralClustering that validates
    parameters and ensures input robustness (e.g., applies scaling
    when negative values are detected).
    """

    def train(self, X_train, y_train=None):
        """Train the spectral clustering model with robustness checks.

        Implements validation and fallback mechanisms to prevent crashes,
        particularly important for negative values and parameter validation.
        """
        logger.debug(
            "[DEBUG] X_train shape: %s, contains NaN: %s", X_train.shape, np.isnan(X_train).any()
        )

        # Verificar e corrigir valores negativos para SpectralClustering
        if np.min(X_train) < 0:
            logger.warning(
                "[Robustness] Input data for SpectralClustering contained negative values. "
                "Applied MinMaxScaler to ensure robustness and avoid model failure."
            )
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            self._scaler = scaler

        # Validar parâmetros para o tamanho do dataset
        n_samples = X_train.shape[0]

        # Ajustar n_neighbors se necessário (para affinity='nearest_neighbors')
        if hasattr(self.model, "n_neighbors") and self.model.n_neighbors is not None:
            if self.model.n_neighbors >= n_samples:
                new_n_neighbors = max(1, n_samples - 1)
                logger.warning(
                    "n_neighbors (%s) >= n_samples (%s). Adjusting to %s",
                    self.model.n_neighbors,
                    n_samples,
                    new_n_neighbors,
                )
                self.model.n_neighbors = new_n_neighbors

        # Validar n_clusters
        if self.model.n_clusters >= n_samples:
            new_n_clusters = max(2, min(3, n_samples // 2))
            logger.warning(
                "n_clusters (%s) >= n_samples (%s). Adjusting to %s",
                self.model.n_clusters,
                n_samples,
                new_n_clusters,
            )
            self.model.n_clusters = new_n_clusters

        try:
            super().train(X_train, y_train)

            # Validar resultado do clustering
            if hasattr(self, "_fitted_labels"):
                n_unique_labels = len(np.unique(self._fitted_labels))
                if n_unique_labels == 1:
                    logger.warning(
                        "SpectralClustering produced only %s cluster. "
                        "This may indicate poor parameter settings or data issues.",
                        n_unique_labels,
                    )
                elif n_unique_labels < self.model.n_clusters:
                    logger.warning(
                        "SpectralClustering produced %s clusters instead of expected %s",
                        n_unique_labels,
                        self.model.n_clusters,
                    )

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            logger.error("SpectralClustering failed with error: %s", e)
            # Fallback: usar KMeans simples como backup
            logger.warning("Falling back to KMeans clustering due to SpectralClustering failure")
            kmeans = KMeans(n_clusters=self.model.n_clusters, random_state=42)
            self._fitted_labels = kmeans.fit_predict(X_train)

    def __init__(self, params: Optional[dict] = None, **kwargs: Any) -> None:
        """Initialize the SpectralClusteringModel."""
        super().__init__()

        # Initialize scaler attribute
        self._scaler = None

        if params is None:
            params = {}
        params.update(kwargs)

        # Validate and sanitize parameters
        params = self._validate_params(params)

        self.model = SpectralClustering(**params)

    def _validate_params(self, params: dict) -> dict:
        """Validate and sanitize SpectralClustering parameters.

        Args:
            params: Raw parameters that may contain invalid values

        Returns:
            dict: Sanitized parameters safe for SpectralClustering
        """
        # Apply default optimizations for large datasets
        params = self._apply_default_optimizations(params)

        # Validate specific parameter types
        params = self._validate_affinity_params(params)
        params = self._validate_assign_labels_params(params)
        params = self._validate_numeric_params(params)

        return params

    def _apply_default_optimizations(self, params: dict) -> dict:
        """Apply default optimizations for large datasets.

        Args:
            params: Original parameters

        Returns:
            dict: Parameters with optimizations applied
        """
        if "affinity" not in params:
            params["affinity"] = "nearest_neighbors"  # More efficient for large datasets

        if "assign_labels" not in params:
            params["assign_labels"] = "discretize"  # Faster than kmeans

        if "n_init" not in params:
            params["n_init"] = 1  # Single initialization for speed

        return params

    def _validate_affinity_params(self, params: dict) -> dict:
        """Validate and sanitize affinity-related parameters.

        Args:
            params: Parameters to validate

        Returns:
            dict: Parameters with valid affinity settings
        """
        valid_affinities = {
            "sigmoid",
            "additive_chi2",
            "laplacian",
            "linear",
            "nearest_neighbors",
            "rbf",
            "cosine",
            "poly",
            "polynomial",
            "precomputed_nearest_neighbors",
            "precomputed",
            "chi2",
        }

        # Sanitize affinity parameter
        if "affinity" in params and params["affinity"] not in valid_affinities:
            logger.warning(
                "Invalid affinity '%s' replaced with 'nearest_neighbors'", params["affinity"]
            )
            params["affinity"] = "nearest_neighbors"

        # Ensure n_neighbors is reasonable for nearest_neighbors affinity
        if params.get("affinity") == "nearest_neighbors":
            if "n_neighbors" not in params:
                params["n_neighbors"] = 10
            elif params["n_neighbors"] < 1:
                params["n_neighbors"] = 10

        return params

    def _validate_assign_labels_params(self, params: dict) -> dict:
        """Validate and sanitize assign_labels parameter.

        Args:
            params: Parameters to validate

        Returns:
            dict: Parameters with valid assign_labels setting
        """
        valid_assign_labels = {"kmeans", "discretize"}

        if "assign_labels" in params and params["assign_labels"] not in valid_assign_labels:
            logger.warning(
                "Invalid assign_labels '%s' replaced with 'discretize'", params["assign_labels"]
            )
            params["assign_labels"] = "discretize"

        return params

    def _validate_numeric_params(self, params: dict) -> dict:
        """Validate and sanitize numeric parameters.

        Args:
            params: Parameters to validate

        Returns:
            dict: Parameters with valid numeric values
        """
        # Ensure n_clusters is positive
        if "n_clusters" in params and params["n_clusters"] < 1:
            params["n_clusters"] = 2

        # Ensure n_init is positive
        if "n_init" in params and params["n_init"] < 1:
            params["n_init"] = 1

        return params
