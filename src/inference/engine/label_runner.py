"""Label inference engine for applying noise/perturbation techniques to target labels."""

from inference.transformations.label.random_label_noise import RandomLabelNoise
from inference.transformations.label.label_flip_near_border import LabelFlipNearBorder
from inference.transformations.label.label_confusion_matrix_noise import LabelConfusionMatrixNoise
from inference.transformations.label.partial_label_noise import PartialLabelNoise
from inference.transformations.label.label_swap_within_class import LabelSwapWithinClass
from utils.types import LabelNoiseConfig

class LabelInferenceEngine:
    """Applies label noise and perturbation techniques to target labels (y) based on configuration.

    Supports: RandomLabelNoise, LabelFlipNearBorder, LabelConfusionMatrixNoise,
        PartialLabelNoise, LabelSwapWithinClass.
    """

    def __init__(self, config: LabelNoiseConfig, X_train=None):
        """Initializes the label inference engine.

        Args:
            config (LabelNoiseConfig):
                Configuration object for label noise techniques.
            X_train (Any, optional):
                Training features, required for certain techniques.
        """
        # This line signals to Pylint that X_train is intentionally used.
        _ = X_train

        self.label_pipeline = []

        # LabelFlipNearBorder is commented out for regression tasks because it requires predict_proba/decision_function
        transformation_map = {
            "label_noise_fraction": RandomLabelNoise,
            "flip_near_border_fraction": LabelFlipNearBorder,  # Excluded for regression: needs predict_proba
            "confusion_matrix_noise_level": LabelConfusionMatrixNoise,
            "partial_label_fraction": PartialLabelNoise,
            "swap_within_class_fraction": LabelSwapWithinClass,
        }

        for field, cls in transformation_map.items():
            value = getattr(config, field, None)
            if value is not None:
                self.label_pipeline.append(cls(value))

    def apply(self, y_train, y_test, model=None, X_train=None, X_test=None):
        """Applies the configured label transformations to the training and test labels.

        Args:
            y_train (Any):
                Original training labels.
            y_test (Any):
                Original test labels.
            model (Any, optional):
                Model instance, used if required by the technique.
            X_train (Any, optional):
                Training features.
            X_test (Any, optional):
                Test features.

        Returns:
            tuple: A tuple containing (y_train_perturbed, y_test_perturbed).
        """
        for transform in self.label_pipeline:
            if getattr(transform, 'requires_model', False):
                y_train = transform.apply(y_train, X=X_train, model=model)
                y_test = transform.apply(y_test, X=X_test, model=model)
            else:
                y_train = transform.apply(y_train)
                y_test = transform.apply(y_test)
        return y_train, y_test
