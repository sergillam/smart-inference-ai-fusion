"""Gradient Boosting Classifier model for the Smart Inference AI Fusion framework.

This module provides a wrapper for sklearn's GradientBoostingClassifier.
"""

import logging
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier

from smart_inference_ai_fusion.core.base_classification_model import BaseClassificationModel

logger = logging.getLogger(__name__)


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

        # 🧪 SCIENTIFIC PROTECTION: Sanitize parameters to prevent fatal errors
        params = self._sanitize_gb_params(params)

        self.model = GradientBoostingClassifier(**params)

    def _sanitize_gb_params(self, params: dict) -> dict:
        """Sanitize GradientBoosting parameters to prevent validation errors.

        Args:
            params (dict): Raw parameters from experiments.

        Returns:
            dict: Sanitized parameters safe for GradientBoostingClassifier.
        """
        sanitized = params.copy()

        # 🧪 SCIENTIFIC PROTECTION: subsample must be in (0.0, 1.0]
        if "subsample" in sanitized:
            value = sanitized["subsample"]
            if not isinstance(value, (int, float)) or value <= 0 or value > 1.0:
                logger.warning(
                    "🧪 SCIENTIFIC PROTECTION: Invalid subsample '%s' -> 0.8 "
                    "(preventing InvalidParameterError)",
                    value,
                )
                sanitized["subsample"] = 0.8  # Safe default value
            elif value > 1.0:
                logger.warning(
                    "🧪 SCIENTIFIC PROTECTION: subsample %.3f > 1.0 -> 1.0 "
                    "(preventing InvalidParameterError)",
                    value,
                )
                sanitized["subsample"] = 1.0

        # 🧪 SCIENTIFIC PROTECTION: learning_rate must be > 0
        if "learning_rate" in sanitized:
            value = sanitized["learning_rate"]
            if not isinstance(value, (int, float)) or value <= 0:
                logger.warning(
                    "🧪 SCIENTIFIC PROTECTION: Invalid learning_rate '%s' -> 0.1 "
                    "(preventing InvalidParameterError)",
                    value,
                )
                sanitized["learning_rate"] = 0.1

        return sanitized
