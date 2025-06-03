from inference.transformations.params.int_noise import IntegerNoise
from inference.transformations.params.bool_flip import BooleanFlip
from inference.transformations.params.str_mutator import StringMutator
from inference.transformations.params.semantic_mutation import SemanticMutation
from inference.transformations.params.scale_hyper import ScaleHyperparameter
from inference.transformations.params.cross_dependency import CrossDependencyPerturbation

class ParameterInferenceEngine:
    """
    Applies perturbation techniques to model hyperparameters.
    Supports both single-key and cross-parameter transformations.
    """
    def __init__(self):
        self.perturber_classes = [
            IntegerNoise,
            BooleanFlip,
            StringMutator,
            SemanticMutation,
            ScaleHyperparameter,
            CrossDependencyPerturbation,
        ]
        self.log = {}

    def apply(self, params: dict) -> dict:
        perturbed = params.copy()

        for key, value in params.items():
            for PerturberClass in self.perturber_classes:
                if PerturberClass.supports(value):
                    perturber = PerturberClass(key)
                    result = perturber.apply(perturbed)

                    # Dict-style response (cross-param updates)
                    if isinstance(result, dict):
                        for updated_key, updated_value in result.items():
                            self.log[updated_key] = {
                                "original": perturbed.get(updated_key),
                                "perturbed": updated_value,
                                "technique": PerturberClass.__name__,
                            }
                            perturbed[updated_key] = updated_value
                    # Single value response
                    elif result is not None:
                        self.log[key] = {
                            "original": perturbed[key],
                            "perturbed": result,
                            "technique": PerturberClass.__name__,
                        }
                        perturbed[key] = result

                    break  # Only one transformation per key
        return perturbed

    def export_log(self) -> dict:
        return self.log
