from inference.transformations.label.random_label_noise import RandomLabelNoise
from inference.transformations.label.label_flip_near_border import LabelFlipNearBorder
from inference.transformations.label.label_confusion_matrix_noise import LabelConfusionMatrixNoise
from inference.transformations.label.partial_label_noise import PartialLabelNoise
from inference.transformations.label.label_swap_within_class import LabelSwapWithinClass
from utils.types import LabelNoiseConfig

class LabelInferenceEngine:
    """
    Aplica técnicas de inferência nos rótulos (y) com base na configuração fornecida.

    Suporta: RandomLabelNoise, LabelFlipNearBorder, LabelConfusionMatrixNoise,
             PartialLabelNoise, LabelSwapWithinClass

    Args:
        config (LabelNoiseConfig): Configuração das técnicas de inferência de rótulo.
        X_train (np.ndarray, opcional): Dados de treino (caso técnica exija X).
    """

    def __init__(self, config: LabelNoiseConfig, X_train=None):
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
                self.label_pipeline.append(cls(value))

    def apply(self, y_train, y_test, model=None, X_train=None, X_test=None):
        """Aplica as transformações definidas na pipeline."""
        for transform in self.label_pipeline:
            if getattr(transform, 'requires_model', False):
                y_train = transform.apply(y_train, X=X_train, model=model)
                y_test = transform.apply(y_test, X=X_test, model=model)
            else:
                y_train = transform.apply(y_train)
                y_test = transform.apply(y_test)
        return y_train, y_test
