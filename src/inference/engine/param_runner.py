"""Parameter inference engine for applying perturbations to model hyperparameters."""

from inference.transformations.params.int_noise import IntegerNoise
from inference.transformations.params.bool_flip import BooleanFlip
from inference.transformations.params.str_mutator import StringMutator
from inference.transformations.params.semantic_mutation import SemanticMutation
from inference.transformations.params.scale_hyper import ScaleHyperparameter
from inference.transformations.params.cross_dependency import CrossDependencyPerturbation
from inference.transformations.params.random_from_space import RandomFromSpace
from inference.transformations.params.bounded_numeric import BoundedNumericShift
from inference.transformations.params.type_cast_perturbation import TypeCastPerturbation
from inference.transformations.params.enum_boundary_shift import EnumBoundaryShift
from utils.types import ParameterNoiseConfig

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

    def apply(self, params: dict) -> dict:
        """Applies enabled perturbation techniques to model hyperparameters.

        Args:
            params (dict): Original model hyperparameters.

        Returns:
            dict: Perturbed model hyperparameters.
        """
        perturbed = params.copy()
        for key, value in params.items():
            for perturber_cls in self.perturber_classes:
                if perturber_cls.supports(value):
                    perturber = perturber_cls(key)
                    result = perturber.apply(perturbed)
                    # Dict-style response (cross-param updates)
                    if isinstance(result, dict):
                        for updated_key, updated_value in result.items():
                            if updated_key not in self.log:
                                self.log[updated_key] = {
                                    "original": perturbed.get(updated_key),
                                    "perturbed": updated_value,
                                    "technique": perturber_cls.__name__,
                                }
                            perturbed[updated_key] = updated_value
                    # Single value response
                    elif result is not None:
                        if key not in self.log:
                            self.log[key] = {
                                "original": perturbed[key],
                                "perturbed": result,
                                "technique": perturber_cls.__name__,
                            }
                        perturbed[key] = result
                    break  # Only one transformation per key
        return perturbed

    def export_log(self) -> dict:
        """Returns a log of all perturbations applied to the parameters.

        Returns:
            dict: Log of parameter changes, keyed by parameter name.
        """
        return self.log
