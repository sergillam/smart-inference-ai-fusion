"""Label transformation that flips labels near the decision border of a model."""

import numpy as np
from inference.transformations.label.base import LabelTransformation

class LabelFlipNearBorder(LabelTransformation):
    """Flips a fraction of labels near the model's decision border.

    This transformation uses model confidence (from `predict_proba` or `decision_function`)
    to identify the least confident samples and flips their labels.

    Attributes:
        flip_fraction (float): Fraction of labels to flip.
        requires_model (bool): Indicates that this transformation requires a model and features.
    """
    requires_model = True

    def __init__(self, flip_fraction):
        """Initializes the transformation.

        Args:
            flip_fraction (float): Fraction of samples to flip (between 0 and 1).
        """
        self.flip_fraction = flip_fraction
        self.requires_model = True

    def _get_confidence(self, model, X):
        """Gets model confidence for each sample using predict_proba or decision_function.

        Args:
            model (object): Trained classifier supporting `predict_proba` or `decision_function`.
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Confidence score for each sample.

        Raises:
            ValueError: If the model does not support any known confidence methods.
        """
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            return np.max(probs, axis=1)
        if hasattr(model, "decision_function"):
            decision = model.decision_function(X)
            return np.abs(decision)
        if hasattr(model, "model"):  # Supports wrapped models
            return self._get_confidence(model.model, X)
        raise ValueError("Model must support predict_proba or decision_function.")

    def apply(self, y, X=None, model=None):
        """Flips labels with the lowest model confidence.

        Args:
            y (Any):
                Original labels.
            X (Any, optional):
                Feature matrix, which is required for this transformation.
            model (Any, optional):
                A trained model, which is required for this transformation.

        Returns:
            np.ndarray: A new label vector with selected labels flipped.

        Raises:
            ValueError: If the `model` or `X` is not provided, or if their
                lengths do not match the length of `y`.
        """
        if model is None or X is None:
            raise ValueError("This transformation requires both a model and X.")
        if len(X) != len(y):
            raise ValueError(
                f"Length mismatch: X has {len(X)} samples, y has {len(y)} labels."
            )

        confidence = self._get_confidence(model, X)
        n = int(len(y) * self.flip_fraction)
        low_conf_indices = np.argsort(confidence)[:n]

        y_flipped = np.array(y).copy()
        classes = np.unique(y)
        for idx in low_conf_indices:
            available = classes[classes != y_flipped[idx]]
            if available.size > 0:
                y_flipped[idx] = np.random.choice(available)

        return y_flipped
