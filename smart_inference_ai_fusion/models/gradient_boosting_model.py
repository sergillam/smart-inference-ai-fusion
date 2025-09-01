"""Gradient Boosting Classifier model for the Smart Inference AI Fusion framework.

This module provides a wrapper for sklearn's GradientBoostingClassifier.
"""

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier

from smart_inference_ai_fusion.core.base_classification_model import BaseClassificationModel


class GradientBoostingModel(BaseClassificationModel):
    """Gradient Boosting Classifier model wrapper for the framework.

    This class wraps sklearn's ``GradientBoostingClassifier`` and exposes a
    consistent interface.
    """

    def __init__(self, params: dict | None = None, **kwargs: Any) -> None:
        """Initialize the GradientBoostingModel.

        Args:
            params (dict | None): Parameters for the ``GradientBoostingClassifier``
                constructor. If ``None``, an empty dict is used.
            **kwargs: Additional keyword arguments for model parameters (merged
                into ``params``).
        """
        super().__init__()
        if params is None:
            params = {}
        params.update(kwargs)
        self.model = GradientBoostingClassifier(**params)
