"""Random Forest Classifier model for the Smart Inference AI Fusion framework.

This module provides a wrapper for sklearn's RandomForestClassifier.
"""

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from smart_inference_ai_fusion.core.base_classification_model import BaseClassificationModel


class RandomForestClassifierModel(BaseClassificationModel):
    """Random Forest Classifier model wrapper for the framework.

    This class wraps sklearn's ``RandomForestClassifier`` and provides a consistent
    interface.
    """

    def __init__(self, params: dict | None = None, **kwargs: Any) -> None:
        """Initialize the RandomForestClassifierModel.

        Args:
            params (dict | None): Parameters for the ``RandomForestClassifier``
                constructor. If ``None``, an empty dict is used.
            **kwargs: Additional keyword arguments for model parameters (merged
                into ``params``).
        """
        super().__init__()
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = RandomForestClassifier(**params)
