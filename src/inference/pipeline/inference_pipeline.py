"""Unified inference pipeline for data, label, and parameter perturbations."""

from inference.engine.inference_engine import InferenceEngine
from inference.engine.label_runner import LabelInferenceEngine
from inference.engine.param_runner import ParameterInferenceEngine


class InferencePipeline:
    """Unified pipeline for applying inference/perturbations.

    This class organizes the application of:
      - Data perturbations (X) via data_noise_config.
      - Label perturbations (y) via label_noise_config.
      - Model hyperparameter perturbations via param_noise_config.

    Attributes:
        data_engine (InferenceEngine or None): Applies data noise techniques.
        label_engine (LabelInferenceEngine or None): Applies label noise techniques.
        param_engine (ParameterInferenceEngine): Applies parameter perturbations.
    """

    def __init__(self, data_noise_config=None, label_noise_config=None, X_train=None):
        """Initializes the inference pipeline.

        Args:
            data_noise_config (object, optional):
                Configuration for data noise (features).
            label_noise_config (object, optional):
                Configuration for label noise.
            X_train (Any, optional):
                Training features, may be required by some label perturbations.
        """
        self.data_engine = (
            InferenceEngine(data_noise_config) if data_noise_config else None
        )
        self.label_engine = (
            LabelInferenceEngine(label_noise_config, X_train=X_train)
            if label_noise_config else None
        )
        self.param_engine = ParameterInferenceEngine()

    def apply_param_inference(self, model_class, base_params, seed=None, ignore_rules=None):
        """Applies inference/perturbation to model hyperparameters.

        Args:
            model_class (type):
                The class of the model to instantiate.
            base_params (dict):
                Base parameters for the model.
            seed (int, optional):
                Reserved for future use.
            ignore_rules (list, optional):
                Reserved for future use.

        Returns:
            tuple: A tuple of `(model, log_dict)` where `model` is the
                instantiated model and `log_dict` contains perturbation details.
        """
        _ = seed, ignore_rules  # Explicitly mark unused arguments
        perturbed_params = self.param_engine.apply(base_params)
        model = model_class(**perturbed_params)
        return model, {"perturbed_params": perturbed_params}

    def apply_data_inference(self, X_train, X_test):
        """Applies data perturbations to input features.

        Args:
            X_train (Any):
                Training features.
            X_test (Any):
                Test features.

        Returns:
            tuple: A tuple of `(X_train_perturbed, X_test_perturbed)`.
        """
        if not self.data_engine:
            return X_train, X_test
        return self.data_engine.apply(X_train, X_test)

    def apply_label_inference(self, y_train, y_test, model=None, X_train=None, X_test=None):
        """Applies label perturbations to target labels.

        Args:
            y_train (Any):
                Training labels.
            y_test (Any):
                Test labels.
            model (Any, optional):
                Model instance, if required by some techniques.
            X_train (Any, optional):
                Training features, used by certain label techniques.
            X_test (Any, optional):
                Test features, used by certain label techniques.

        Returns:
            tuple: A tuple of `(y_train_perturbed, y_test_perturbed)`.
        """
        if not self.label_engine:
            return y_train, y_test
        return self.label_engine.apply(
            y_train=y_train, y_test=y_test, model=model, X_train=X_train, X_test=X_test
        )
