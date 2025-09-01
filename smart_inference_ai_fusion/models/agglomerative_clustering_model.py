"""Agglomerative Clustering model for the Smart Inference AI Fusion framework.

This module defines the AgglomerativeClusteringModel class, a wrapper for
scikit-learn's AgglomerativeClustering compatible with the BaseModel interface.
"""

from typing import Any, Optional

from sklearn.cluster import AgglomerativeClustering

from smart_inference_ai_fusion.core.base_clustering_model import BaseClusteringModel


class AgglomerativeClusteringModel(BaseClusteringModel):
    """Wrapper for scikit-learn's AgglomerativeClustering, compatible with BaseModel."""

    def __init__(self, params: Optional[dict] = None, **kwargs: Any) -> None:
        """Initialize the AgglomerativeClusteringModel.

        Args:
            params (dict | None): Parameters for ``AgglomerativeClustering``.
            **kwargs: Additional keyword arguments merged into ``params``.
        """
        super().__init__()
        if params is None:
            params = {}
        params.update(kwargs)
        # Enforce standard usage: use n_clusters (not distance_threshold)
        if "distance_threshold" in params and params["distance_threshold"] is not None:
            # If user passes distance_threshold, ensure n_clusters is None
            params["n_clusters"] = None
        self.model = AgglomerativeClustering(**params)
