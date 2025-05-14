from inference.transformations.label.label_noise import LabelNoise
from utils.types import DatasetNoiseConfig 

class LabelInferenceEngine:
    """
    Executa técnicas de inferência aplicadas aos rótulos (y), como Label Noise.

    Args:
        config (DatasetNoiseConfig): Configuração com possíveis parâmetros de ruído para rótulos.
    """
    def __init__(self, config: DatasetNoiseConfig):
        self.label_pipeline = []

        if config.label_noise_fraction is not None:
            self.label_pipeline.append(LabelNoise(config.label_noise_fraction))

    def apply(self, y_train, y_test):
        for transform in self.label_pipeline:
            y_train = transform.apply(y_train)
            y_test = transform.apply(y_test)
        return y_train, y_test
