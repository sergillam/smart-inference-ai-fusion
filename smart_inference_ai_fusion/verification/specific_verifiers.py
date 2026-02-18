"""Verificações específicas para dados, labels e parâmetros."""

import logging
from typing import Any, Dict

from .core.plugin_interface import VerificationInput, VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


class DataIntegrityVerifier:
    """Verificador específico para integridade de dados."""

    @staticmethod
    def verify_shape_preservation(input_data: Any, output_data: Any) -> VerificationResult:
        """Verifica se as dimensões dos dados foram preservadas."""
        try:
            # Verificar se ambos os dados têm atributo shape
            if not (hasattr(input_data, "shape") and hasattr(output_data, "shape")):
                return VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    verifier_name="DataIntegrityVerifier",
                    execution_time=0.0,
                    message="Data doesn't have shape attribute",
                )

            if input_data.shape != output_data.shape:
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    verifier_name="DataIntegrityVerifier",
                    execution_time=0.0,
                    message=f"Shape mismatch: {input_data.shape} != {output_data.shape}",
                )

            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                verifier_name="DataIntegrityVerifier",
                execution_time=0.0,
                message="Shape preserved successfully",
            )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verifier_name="DataIntegrityVerifier",
                execution_time=0.0,
                message=f"Error verifying shape: {str(e)}",
            )

    @staticmethod
    def verify_bounds_preservation(
        input_data: Any, output_data: Any, tolerance: float = 0.1
    ) -> VerificationResult:
        """Verifica se os bounds dos dados foram preservados."""
        try:
            if not (hasattr(input_data, "min") and hasattr(input_data, "max")):
                return VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    verifier_name="DataIntegrityVerifier",
                    execution_time=0.0,
                    message="Data doesn't support min/max operations",
                )

            input_min, input_max = float(input_data.min()), float(input_data.max())
            output_min, output_max = float(output_data.min()), float(output_data.max())

            # Verificar se os bounds estão dentro da tolerância
            min_diff = abs(output_min - input_min) / (abs(input_min) + 1e-8)
            max_diff = abs(output_max - input_max) / (abs(input_max) + 1e-8)

            if min_diff > tolerance or max_diff > tolerance:
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    verifier_name="DataIntegrityVerifier",
                    execution_time=0.0,
                    message=f"Bounds changed beyond tolerance: min_diff={min_diff:.4f}, max_diff={max_diff:.4f}",
                )

            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                verifier_name="DataIntegrityVerifier",
                execution_time=0.0,
                message="Bounds preserved within tolerance",
            )

        except TypeError:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                verifier_name="DataIntegrityVerifier",
                execution_time=0.0,
                message="Data type does not support min/max operations",
            )
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verifier_name="DataIntegrityVerifier",
                execution_time=0.0,
                message=f"Error verifying bounds: {str(e)}",
            )


class LabelIntegrityVerifier:
    """Verificador específico para integridade de labels."""

    @staticmethod
    def verify_class_distribution(
        input_labels: Any, output_labels: Any, tolerance: float = 0.05
    ) -> VerificationResult:
        """Verifica se a distribuição de classes foi preservada."""
        try:
            import numpy as np

            # Converter para arrays numpy se necessário
            input_array = (
                np.array(input_labels) if not isinstance(input_labels, np.ndarray) else input_labels
            )
            output_array = (
                np.array(output_labels)
                if not isinstance(output_labels, np.ndarray)
                else output_labels
            )

            # Calcular distribuições
            input_unique, input_counts = np.unique(input_array, return_counts=True)
            output_unique, output_counts = np.unique(output_array, return_counts=True)

            # Verificar se as classes são as mesmas
            if not np.array_equal(np.sort(input_unique), np.sort(output_unique)):
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    verifier_name="LabelIntegrityVerifier",
                    execution_time=0.0,
                    message=f"Class sets differ: {input_unique} vs {output_unique}",
                )

            # Verificar distribuições
            input_dist = input_counts / len(input_array)
            output_dist = output_counts / len(output_array)

            max_diff = np.max(np.abs(input_dist - output_dist))

            if max_diff > tolerance:
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    verifier_name="LabelIntegrityVerifier",
                    execution_time=0.0,
                    message=f"Class distribution changed beyond tolerance: max_diff={max_diff:.4f}",
                )

            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                verifier_name="LabelIntegrityVerifier",
                execution_time=0.0,
                message="Class distribution preserved within tolerance",
            )

        except ImportError:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                verifier_name="LabelIntegrityVerifier",
                execution_time=0.0,
                message="NumPy not available for distribution verification",
            )
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verifier_name="LabelIntegrityVerifier",
                execution_time=0.0,
                message=f"Error verifying class distribution: {str(e)}",
            )


class ParameterIntegrityVerifier:
    """Verificador específico para integridade de parâmetros."""

    @staticmethod
    def verify_parameter_types(
        input_params: Dict[str, Any], output_params: Dict[str, Any]
    ) -> VerificationResult:
        """Verifica se os tipos dos parâmetros foram preservados."""
        try:
            type_mismatches = []

            for key in input_params:
                if key in output_params:
                    input_type = type(input_params[key])
                    output_type = type(output_params[key])

                    if input_type != output_type:
                        type_mismatches.append(
                            f"{key}: {input_type.__name__} -> {output_type.__name__}"
                        )

            if type_mismatches:
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    verifier_name="ParameterIntegrityVerifier",
                    execution_time=0.0,
                    message=f"Parameter type mismatches: {', '.join(type_mismatches)}",
                )

            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                verifier_name="ParameterIntegrityVerifier",
                execution_time=0.0,
                message="Parameter types preserved",
            )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verifier_name="ParameterIntegrityVerifier",
                execution_time=0.0,
                message=f"Error verifying parameter types: {str(e)}",
            )

    @staticmethod
    def verify_parameter_bounds(
        params: Dict[str, Any], bounds: Dict[str, Dict[str, Any]]
    ) -> VerificationResult:
        """Verifica se os parâmetros estão dentro dos bounds especificados."""
        try:
            bound_violations = []

            for param_name, param_value in params.items():
                if param_name in bounds:
                    param_bounds = bounds[param_name]

                    # Verificar bound mínimo
                    if "min" in param_bounds and param_value < param_bounds["min"]:
                        bound_violations.append(
                            f"{param_name} < {param_bounds['min']} (value: {param_value})"
                        )

                    # Verificar bound máximo
                    if "max" in param_bounds and param_value > param_bounds["max"]:
                        bound_violations.append(
                            f"{param_name} > {param_bounds['max']} (value: {param_value})"
                        )

                    # Verificar valores válidos
                    if (
                        "valid_values" in param_bounds
                        and param_value not in param_bounds["valid_values"]
                    ):
                        bound_violations.append(
                            f"{param_name} not in {param_bounds['valid_values']} (value: {param_value})"
                        )

            if bound_violations:
                return VerificationResult(
                    status=VerificationStatus.FAILURE,
                    verifier_name="ParameterIntegrityVerifier",
                    execution_time=0.0,
                    message=f"Parameter bound violations: {', '.join(bound_violations)}",
                )

            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                verifier_name="ParameterIntegrityVerifier",
                execution_time=0.0,
                message="All parameters within bounds",
            )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verifier_name="ParameterIntegrityVerifier",
                execution_time=0.0,
                message=f"Error verifying parameter bounds: {str(e)}",
            )


class TransformationVerifier:
    """Verificador para propriedades específicas de transformações."""

    @staticmethod
    def verify_outlier_injection_bounds(
        input_data: Any, output_data: Any, max_outlier_ratio: float = 0.1
    ) -> VerificationResult:
        """Verifica se a injeção de outliers está dentro dos limites."""
        try:
            import numpy as np

            # Calcular diferenças
            if hasattr(input_data, "shape") and hasattr(output_data, "shape"):
                if input_data.shape != output_data.shape:
                    return VerificationResult(
                        status=VerificationStatus.FAILURE,
                        verifier_name="TransformationVerifier",
                        execution_time=0.0,
                        message="Shape mismatch in outlier injection",
                    )

                # Estimar quantos pontos foram modificados significativamente
                diff = np.abs(output_data - input_data)
                threshold = 3 * np.std(input_data)  # Considerar outliers como 3 desvios padrão
                outlier_mask = diff > threshold
                outlier_ratio = np.sum(outlier_mask) / input_data.size

                if outlier_ratio > max_outlier_ratio:
                    return VerificationResult(
                        status=VerificationStatus.FAILURE,
                        verifier_name="TransformationVerifier",
                        execution_time=0.0,
                        message=f"Outlier ratio {outlier_ratio:.4f} exceeds maximum {max_outlier_ratio}",
                    )

            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                verifier_name="TransformationVerifier",
                execution_time=0.0,
                message="Outlier injection within acceptable bounds",
            )

        except ImportError:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                verifier_name="TransformationVerifier",
                execution_time=0.0,
                message="NumPy not available for outlier verification",
            )
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verifier_name="TransformationVerifier",
                execution_time=0.0,
                message=f"Error verifying outlier injection: {str(e)}",
            )


# Registry de verificadores específicos
SPECIFIC_VERIFIERS = {
    "data_integrity": DataIntegrityVerifier,
    "label_integrity": LabelIntegrityVerifier,
    "parameter_integrity": ParameterIntegrityVerifier,
    "transformation": TransformationVerifier,
}


def run_specific_verification(
    verification_type: str, verification_input: VerificationInput
) -> VerificationResult:
    """Executa uma verificação específica baseada no tipo."""
    if verification_type not in SPECIFIC_VERIFIERS:
        return VerificationResult(
            status=VerificationStatus.ERROR,
            verifier_name="SpecificVerifier",
            execution_time=0.0,
            message=f"Unknown verification type: {verification_type}",
        )

    verifier_class = SPECIFIC_VERIFIERS[verification_type]
    constraints = verification_input.constraints

    try:
        # Executar verificação baseada nos constraints
        if verification_type == "data_integrity":
            if constraints.get("preserve_shape"):
                return verifier_class.verify_shape_preservation(
                    verification_input.input_data, verification_input.output_data
                )
            elif constraints.get("preserve_bounds"):
                tolerance = constraints.get("bounds_tolerance", 0.1)
                return verifier_class.verify_bounds_preservation(
                    verification_input.input_data, verification_input.output_data, tolerance
                )

        elif verification_type == "label_integrity":
            if constraints.get("preserve_class_distribution"):
                tolerance = constraints.get("distribution_tolerance", 0.05)
                return verifier_class.verify_class_distribution(
                    verification_input.input_data, verification_input.output_data, tolerance
                )

        elif verification_type == "parameter_integrity":
            if constraints.get("preserve_parameter_types"):
                return verifier_class.verify_parameter_types(
                    verification_input.input_data, verification_input.output_data
                )
            elif constraints.get("parameter_bounds_check"):
                bounds = constraints.get("parameter_bounds", {})
                return verifier_class.verify_parameter_bounds(
                    verification_input.output_data, bounds
                )

        elif verification_type == "transformation":
            if constraints.get("outlier_ratio_bounds"):
                max_ratio = constraints["outlier_ratio_bounds"][1]
                return verifier_class.verify_outlier_injection_bounds(
                    verification_input.input_data, verification_input.output_data, max_ratio
                )

        return VerificationResult(
            status=VerificationStatus.SKIPPED,
            verifier_name="SpecificVerifier",
            execution_time=0.0,
            message=f"No specific verification method for constraints: {constraints}",
        )

    except Exception as e:
        return VerificationResult(
            status=VerificationStatus.ERROR,
            verifier_name="SpecificVerifier",
            execution_time=0.0,
            message=f"Error in specific verification: {str(e)}",
        )
