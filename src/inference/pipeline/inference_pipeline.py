from inference.engine.inference_engine import InferenceEngine
from inference.engine.label_runner import LabelInferenceEngine
from inference.engine.param_runner import ParameterInferenceEngine

class InferencePipeline:
    """
    Unified pipeline that applies inference to:
    - Input data (X)            [data_noise_config]
    - Labels (y)                [label_noise_config]
    - Model hyperparameters     [param_noise_config]
    Allows better organization and reuse of framework components.
    """
    def __init__(self, data_noise_config=None, label_noise_config=None, X_train=None):
        self.data_engine = InferenceEngine(data_noise_config) if data_noise_config else None
        self.label_engine = LabelInferenceEngine(label_noise_config, X_train=X_train) if label_noise_config else None
        self.param_engine = ParameterInferenceEngine()

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
        if not self.data_engine:
            return X_train, X_test
        return self.data_engine.apply(X_train, X_test)

    def apply_label_inference(self, y_train, y_test, model=None, X_train=None, X_test=None):
        if not self.label_engine:
            return y_train, y_test
        return self.label_engine.apply(y_train, y_test, model=model, X_train=X_train, X_test=X_test)
