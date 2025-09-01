"""MLPModel module.

This module defines the MLPModel class, a wrapper for scikit-learn's
MLPClassifier to be used within the smart-inference-ai-fusion framework.
"""

from typing import Any, Optional

from sklearn.neural_network import MLPClassifier

from smart_inference_ai_fusion.core.base_classification_model import BaseClassificationModel
from smart_inference_ai_fusion.models.common import initialize_model_params


class MLPModel(BaseClassificationModel):
    """MLP Classifier model wrapper for the Smart Inference AI Fusion framework.

    This class wraps sklearn's MLPClassifier and exposes a consistent interface.
    """

    def __init__(self, params: Optional[dict] = None, **kwargs: Any) -> None:
        """Initialize the MLPModel.

        Args:
            params (dict | None): Parameters for the ``MLPClassifier`` constructor.
            **kwargs: Additional keyword arguments for model parameters
                (merged into ``params``).
        """
        super().__init__()
        model_params = initialize_model_params(params, **kwargs)
        self.model = MLPClassifier(**model_params)
