"""Parameter inference engine for applying perturbations to model hyperparameters."""

from smart_inference_ai_fusion.inference.transformations.params import (
    BooleanFlip,
    BoundedNumericShift,
    CrossDependencyPerturbation,
    EnumBoundaryShift,
    IntegerNoise,
    RandomFromSpace,
    ScaleHyperparameter,
    SemanticMutation,
    StringMutator,
    TypeCastPerturbation,
)
from smart_inference_ai_fusion.utils.types import ParameterNoiseConfig


class ParameterInferenceEngine:
    """Engine for applying configurable perturbation techniques to model hyperparameters.

    This engine supports a modular set of parameter perturbation techniques,
    which can be enabled or disabled through a ParameterNoiseConfig.

    Attributes:
        config (ParameterNoiseConfig or None): Configuration object specifying enabled techniques.
        log (dict): Records details of all perturbations applied.
        perturber_classes (list): List of enabled perturbation classes.
    """

    def __init__(self, config: ParameterNoiseConfig = None):
        """Initializes the parameter inference engine.

        Args:
            config (ParameterNoiseConfig, optional): Configuration enabling/disabling techniques.
                If None, all techniques are enabled by default.
        """
        self.config = config
        self.log = {}
        self.perturber_classes = []

        mapping = {
            "integer_noise": IntegerNoise,
            "boolean_flip": BooleanFlip,
            "string_mutator": StringMutator,
            "semantic_mutation": SemanticMutation,
            "scale_hyper": ScaleHyperparameter,
            "cross_dependency": CrossDependencyPerturbation,
            "random_from_space": RandomFromSpace,
            "bounded_numeric": BoundedNumericShift,
            "type_cast_perturbation": TypeCastPerturbation,
            "enum_boundary_shift": EnumBoundaryShift,
        }

        if config is None:
            # By default, enable all techniques
            self.perturber_classes = list(mapping.values())
        else:
            for key, cls in mapping.items():
                # Only enable techniques explicitly set to True
                if getattr(config, key, None) is True:
                    self.perturber_classes.append(cls)

    def _log_and_update_dict(self, result: dict, perturbed: dict, technique: str) -> None:
        """Handles logging and updates for a dictionary-based perturbation result."""
        for key, value in result.items():
            if key not in self.log:
                self.log[key] = {
                    "original": perturbed.get(key),
                    "perturbed": value,
                    "technique": technique,
                }
            perturbed[key] = value

    def _log_and_update_single(
        self, result: any, key: str, perturbed: dict, technique: str
    ) -> None:
        """Handles logging and updates for a single-value perturbation result."""
        if key not in self.log:
            self.log[key] = {
                "original": perturbed[key],
                "perturbed": result,
                "technique": technique,
            }
        perturbed[key] = result

    def _apply_perturber_for_key(self, key: str, value: any, perturbed: dict) -> None:
        """Finds and applies the first supported perturber for a given parameter."""
        for perturber_cls in self.perturber_classes:
            if perturber_cls.supports(value):
                perturber = perturber_cls(key)
                result = perturber.apply(perturbed)
                technique = perturber_cls.__name__

                if isinstance(result, dict):
                    self._log_and_update_dict(result, perturbed, technique)
                elif result is not None:
                    self._log_and_update_single(result, key, perturbed, technique)

                break  # Apply only one transformation per key

    def apply(self, params: dict) -> dict:
        """Applies enabled perturbation techniques to model hyperparameters.

        Args:
            params (dict): Original model hyperparameters.

        Returns:
            dict: Perturbed model hyperparameters.
        """
        perturbed = params.copy()
        for key, value in params.items():
            self._apply_perturber_for_key(key, value, perturbed)
        return perturbed

    def export_log(self) -> dict:
        """Returns a log of all perturbations applied to the parameters.

        Returns:
            dict: Log of parameter changes, keyed by parameter name.
        """
        return self.log
