from inference.engine.inference_engine import InferenceEngine
from inference.engine.label_runner import LabelInferenceEngine
from inference.engine.param_runner import apply_param_inference


class InferencePipeline:
    """
    Pipeline unificada que aplica inferência:
    - Nos dados de entrada (X)
    - Nos rótulos (y)
    - Nos hiperparâmetros do modelo
    
    Permite maior organização e reutilização dos componentes do framework.
    """
    def __init__(self, dataset_noise_config: dict):
        self.dataset_noise_config = dataset_noise_config
        self.data_engine = InferenceEngine(dataset_noise_config)
        self.label_engine = LabelInferenceEngine(dataset_noise_config)

    def apply_param_inference(self, model_class, base_params, seed=None, ignore_rules=None):
        """
        Aplica inferência nos parâmetros do modelo.

        Retorna: modelo instanciado com parâmetros perturbados e log
        """
        return apply_param_inference(
            model_class=model_class,
            base_params=base_params,
            seed=seed,
            ignore_rules=ignore_rules
        )

    def apply_data_inference(self, X_train, X_test):
        """
        Aplica as transformações no conjunto de atributos (X).
        """
        return self.data_engine.apply(X_train, X_test)

    def apply_label_inference(self, y_train, y_test):
        """
        Aplica as transformações no conjunto de rótulos (y).
        """
        return self.label_engine.apply(y_train, y_test)
