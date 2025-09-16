"""Unified inference pipeline for data, label, and parameter perturbations."""

import logging
from typing import Dict, Any, Optional, Tuple

from smart_inference_ai_fusion.inference.engine.inference_engine import InferenceEngine
from smart_inference_ai_fusion.inference.engine.label_runner import LabelInferenceEngine
from smart_inference_ai_fusion.inference.engine.param_runner import ParameterInferenceEngine
from smart_inference_ai_fusion.verification.core.formal_verification import verification_manager
from smart_inference_ai_fusion.verification.decorators import verify_pipeline_step
from smart_inference_ai_fusion.utils.verification_report import report_verification_results

logger = logging.getLogger(__name__)


# pylint: disable=too-many-positional-arguments
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

    def __init__(self, data_noise_config=None, label_noise_config=None, X_train=None, 
                 verification_config=None):
        """Initializes the inference pipeline.

        Args:
            data_noise_config (object, optional):
                Configuration for data noise (features).
            label_noise_config (object, optional):
                Configuration for label noise.
            X_train (Any, optional):
                Training features, may be required by some label perturbations.
            verification_config (dict, optional):
                Configuration for formal verification.
        """
        self.data_engine = InferenceEngine(data_noise_config) if data_noise_config else None
        self.label_engine = (
            LabelInferenceEngine(label_noise_config, X_train=X_train)
            if label_noise_config
            else None
        )
        self.param_engine = ParameterInferenceEngine()
        
        # Configuração de verificação
        self.verification_config = verification_config or {}
        self.verification_enabled = self.verification_config.get('enabled', True)
        self.verification_timeout = self.verification_config.get('timeout', 30.0)
        self.fail_on_verification_error = self.verification_config.get('fail_on_error', False)

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
        
        # Aplicar verificação de parâmetros antes da perturbação
        if self.verification_enabled:
            self._verify_parameters("pre_perturbation", base_params)
        
        perturbed_params = self.param_engine.apply(base_params)
        
        # Aplicar verificação de parâmetros após a perturbação
        if self.verification_enabled:
            self._verify_parameters("post_perturbation", perturbed_params, base_params)
        
        model = model_class(**perturbed_params)
        
        # Verificar integridade do modelo criado
        if self.verification_enabled:
            self._verify_model_integrity(model, perturbed_params)
        
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
        
        # Verificar integridade dos dados de entrada
        if self.verification_enabled:
            self._verify_data_integrity("input_data", X_train, X_test)
        
        X_train_perturbed, X_test_perturbed = self.data_engine.apply(X_train, X_test)
        
        # Verificar integridade dos dados após perturbação
        if self.verification_enabled:
            self._verify_data_integrity("output_data", X_train_perturbed, X_test_perturbed, 
                                       original_train=X_train, original_test=X_test)
        
        return X_train_perturbed, X_test_perturbed

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
        
        # Verificar integridade dos labels de entrada
        if self.verification_enabled:
            self._verify_label_integrity("input_labels", y_train, y_test)
        
        y_train_perturbed, y_test_perturbed = self.label_engine.apply(
            y_train=y_train, y_test=y_test, model=model, X_train=X_train, X_test=X_test
        )
        
        # Verificar integridade dos labels após perturbação
        if self.verification_enabled:
            self._verify_label_integrity("output_labels", y_train_perturbed, y_test_perturbed,
                                        original_train=y_train, original_test=y_test)
        
        return y_train_perturbed, y_test_perturbed
    
    def _verify_data_integrity(self, step_name: str, X_train, X_test, 
                              original_train=None, original_test=None):
        """Verifica integridade dos dados."""
        # Mapear para constraints compatíveis com Z3
        constraints = {
            'shape_preservation': True,  # Compatible with Z3
            'type_safety': True,         # Compatible with Z3
            'bounds': True,              # Compatible with Z3
            'range_check': True,         # Compatible with Z3
        }
        
        if original_train is not None and original_test is not None:
            constraints.update({
                'bounds': True,          # Preserve bounds during transformation
                'non_negative': True     # Ensure non-negative values if applicable
            })
        
        verification_result = verification_manager.verify(
            name=f"pipeline.data.{step_name}",
            constraints=constraints,
            input_data={'train': original_train, 'test': original_test} if original_train is not None else None,
            output_data={'train': X_train, 'test': X_test},
            timeout=self.verification_timeout
        )
        
        # Reportar resultados detalhados
        if verification_result:
            report_verification_results(
                verification_result=verification_result,
                model_name="DataPipeline",
                dataset_name="unknown",
                transformation_name=f"data_integrity_{step_name}"
            )
        
        if self.fail_on_verification_error and verification_result.status.value == 'FAILED':
            raise RuntimeError(f"Data verification failed at {step_name}: {verification_result.message}")
    
    def _verify_label_integrity(self, step_name: str, y_train, y_test,
                               original_train=None, original_test=None):
        """Verifica integridade dos labels."""
        # Mapear para constraints compatíveis com Z3
        constraints = {
            'shape_preservation': True,  # Compatible with Z3
            'type_safety': True,         # Compatible with Z3
            'bounds': True,              # Compatible with Z3
            'range_check': True,         # Compatible with Z3
        }
        
        if original_train is not None and original_test is not None:
            constraints.update({
                'bounds': True,          # Preserve bounds during transformation
                'integer_arithmetic': True  # Labels are typically integers
            })
        
        verification_result = verification_manager.verify(
            name=f"pipeline.labels.{step_name}",
            constraints=constraints,
            input_data={'train': original_train, 'test': original_test} if original_train is not None else None,
            output_data={'train': y_train, 'test': y_test},
            timeout=self.verification_timeout
        )
        
        # Reportar resultados detalhados
        if verification_result:
            report_verification_results(
                verification_result=verification_result,
                model_name="LabelPipeline",
                dataset_name="unknown",
                transformation_name=f"label_integrity_{step_name}"
            )
        
        if self.fail_on_verification_error and verification_result.status.value == 'FAILED':
            raise RuntimeError(f"Label verification failed at {step_name}: {verification_result.message}")
    
    def _verify_parameters(self, step_name: str, params: Dict[str, Any], 
                          original_params: Optional[Dict[str, Any]] = None):
        """Verifica integridade dos parâmetros."""
        # Mapear para constraints compatíveis com Z3
        constraints = {
            'type_safety': True,         # Compatible with Z3
            'bounds': True,              # Compatible with Z3
            'range_check': True,         # Compatible with Z3
            'real_arithmetic': True,     # For numeric parameters
        }
        
        if original_params is not None:
            constraints.update({
                'bounds': True,          # Preserve parameter bounds
                'type_safety': True      # Maintain parameter types
            })
        
        verification_result = verification_manager.verify(
            name=f"pipeline.parameters.{step_name}",
            constraints=constraints,
            input_data=original_params,
            output_data=params,
            timeout=self.verification_timeout
        )
        
        # Reportar resultados detalhados
        if verification_result:
            report_verification_results(
                verification_result=verification_result,
                model_name="ParameterPipeline",
                dataset_name="unknown",
                transformation_name=f"parameter_integrity_{step_name}"
            )
        
        if self.fail_on_verification_error and verification_result.status.value == 'FAILED':
            raise RuntimeError(f"Parameter verification failed at {step_name}: {verification_result.message}")
    
    def _verify_model_integrity(self, model, params: Dict[str, Any]):
        """Verifica integridade do modelo criado."""
        # Mapear para constraints compatíveis com Z3
        constraints = {
            'type_safety': True,         # Compatible with Z3
            'bounds': True,              # Compatible with Z3
            'range_check': True,         # Compatible with Z3
        }
        
        verification_result = verification_manager.verify(
            name=f"pipeline.model.{model.__class__.__name__}",
            constraints=constraints,
            input_data=params,
            output_data=model,
            timeout=self.verification_timeout
        )
        
        if self.fail_on_verification_error and verification_result.status.value == 'FAILED':
            raise RuntimeError(f"Model verification failed: {verification_result.message}")
