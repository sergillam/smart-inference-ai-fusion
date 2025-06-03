from inference.engine.inference_engine import InferenceEngine
from inference.engine.label_runner import LabelInferenceEngine
from inference.engine.param_runner import ParameterInferenceEngine  # ✅ novo


class InferencePipeline:
    """
    Unified pipeline that applies inference to:
    - Input data (X)
    - Labels (y)
    - Model hyperparameters

    Allows better organization and reuse of framework components.
    """
    def __init__(self, dataset_noise_config: dict):
        self.dataset_noise_config = dataset_noise_config
        self.data_engine = InferenceEngine(dataset_noise_config)
        self.label_engine = LabelInferenceEngine(dataset_noise_config)
        self.param_engine = ParameterInferenceEngine()  # ✅ engine local da pipeline

    def apply_param_inference(self, model_class, base_params, seed=None, ignore_rules=None):
        """
        Applies inference to the model's parameters.

        Returns: model instantiated with perturbed parameters and log.
        """
        perturbed_params = self.param_engine.apply(base_params)
        model = model_class(**perturbed_params)
        return model, {"perturbed_params": perturbed_params}

    def apply_data_inference(self, X_train, X_test):
        """
        Applies transformations to the input features (X).
        """
        return self.data_engine.apply(X_train, X_test)

    def apply_label_inference(self, y_train, y_test):
        """
        Applies transformations to the labels (y).
        """
        return self.label_engine.apply(y_train, y_test)
