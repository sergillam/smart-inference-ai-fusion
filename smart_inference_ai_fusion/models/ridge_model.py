"""RidgeModel module for the Smart Inference AI Fusion framework.

This module defines the RidgeModel class, a wrapper for scikit-learn's
RidgeClassifier to be used within the framework.
"""

from typing import Any, Optional

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier

from smart_inference_ai_fusion.core.base_classification_model import BaseClassificationModel
from smart_inference_ai_fusion.models.common import initialize_model_params


class RidgeModel(BaseClassificationModel):
    """Ridge Classifier model wrapper for the Smart Inference AI Fusion framework.

    This class wraps sklearn's ``RidgeClassifier`` and exposes a consistent interface.
    """

    def __init__(self, params: Optional[dict] = None, **kwargs: Any) -> None:
        """Initialize the RidgeModel.

        Args:
            params (dict | None): Parameters for the ``RidgeClassifier`` constructor.
                If ``None``, an empty dict is used.
            **kwargs: Additional keyword arguments merged into ``params``.
        """
        super().__init__()
        model_params = initialize_model_params(params, **kwargs)
        self.model = RidgeClassifier(**model_params)

    def predict_proba(self, X: Any):
        """Approximate class probabilities using ``decision_function`` + softmax.

        Args:
            X: Input features. Shape: ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Pseudo-probabilities. Shape: ``(n_samples, n_classes)``.

        Raises:
            AttributeError: If the underlying estimator lacks ``decision_function``.
        """
        if not hasattr(self.model, "decision_function"):
            raise AttributeError("RidgeClassifier does not support decision_function.")
        scores = self.model.decision_function(X)
        # Binary case: shape (n_samples,) → make it (n_samples, 2)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        probs = softmax(scores, axis=1)
        return probs
