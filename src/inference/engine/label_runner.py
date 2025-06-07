from inference.transformations.label.random_label_noise import RandomLabelNoise
from inference.transformations.label.label_flip_near_border import LabelFlipNearBorder
from inference.transformations.label.label_confusion_matrix_noise import LabelConfusionMatrixNoise
from inference.transformations.label.partial_label_noise import PartialLabelNoise
from inference.transformations.label.label_swap_within_class import LabelSwapWithinClass
from utils.types import DatasetNoiseConfig


class LabelInferenceEngine:
    """
    Aplica técnicas de inferência nos rótulos (y) com base na configuração fornecida.

    Suporta: RandomLabelNoise, LabelFlipNearBorder, LabelConfusionMatrixNoise,
             PartialLabelNoise, LabelSwapWithinClass
    """

    def __init__(self, config: DatasetNoiseConfig, X_train=None):
        self.label_pipeline = []

        transformation_map = {
            "label_noise_fraction": RandomLabelNoise,
            "flip_near_border_fraction": LabelFlipNearBorder,
            "confusion_matrix_noise_level": LabelConfusionMatrixNoise,
            "partial_label_fraction": PartialLabelNoise,
            "swap_within_class_fraction": LabelSwapWithinClass,
        }

        for field, cls in transformation_map.items():
            value = getattr(config, field, None)
            if value is not None:
                if cls == LabelFlipNearBorder:
                    self.label_pipeline.append(cls(value, X_train))
                else:
                    self.label_pipeline.append(cls(value))

    def apply(self, y_train, y_test):
        for transform in self.label_pipeline:
            y_train = transform.apply(y_train)
            y_test = transform.apply(y_test)
        return y_train, y_test
