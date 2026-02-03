"""Decoradores para integração de verificação formal."""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from .core.formal_verification import verification_manager
from .core.plugin_interface import VerificationStatus

logger = logging.getLogger(__name__)


def verify_transformation(
    constraints: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
    timeout: float = 30.0,
    verifier_name: Optional[str] = None,
    fail_on_error: bool = False,
):
    """
    Decorador para aplicar verificação formal em transformações.

    Args:
        constraints: Constraints específicos para verificação
        enabled: Se a verificação está habilitada
        timeout: Timeout para verificação
        verifier_name: Nome específico do verificador
        fail_on_error: Se deve falhar quando verificação encontra erro
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extrair dados relevantes dos argumentos
            input_data = args[0] if args else None

            # Executar função original
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Aplicar verificação se habilitada
            if enabled and verification_manager.enabled:
                # Preparar constraints padrão se não fornecidos
                verification_constraints = constraints or {}

                # Adicionar constraints automáticos baseados no tipo de transformação
                if not verification_constraints:
                    verification_constraints = _infer_constraints(func.__name__, input_data, result)

                # Executar verificação
                verification_result = verification_manager.verify(
                    name=f"{func.__module__}.{func.__name__}",
                    constraints=verification_constraints,
                    input_data=input_data,
                    output_data=result,
                    parameters=kwargs,
                    timeout=timeout,
                    verifier_name=verifier_name,
                )

                # Log do resultado
                logger.info(f"Verification for {func.__name__}: {verification_result.status.value}")
                if verification_result.message:
                    logger.debug(f"Verification message: {verification_result.message}")

                # Falhar se necessário
                if fail_on_error and verification_result.status == VerificationStatus.FAILED:
                    raise RuntimeError(
                        f"Verification failed for {func.__name__}: {verification_result.message}"
                    )

                # Adicionar resultado da verificação aos metadados do resultado
                if hasattr(result, "__dict__"):
                    result.__verification_result__ = verification_result

            return result

        # Marcar função como verificada
        wrapper.__verified__ = True
        wrapper.__verification_config__ = {
            "constraints": constraints,
            "enabled": enabled,
            "timeout": timeout,
            "verifier_name": verifier_name,
            "fail_on_error": fail_on_error,
        }

        return wrapper

    return decorator


def _infer_constraints(func_name: str, input_data: Any, output_data: Any) -> Dict[str, Any]:
    """Infere constraints automáticos baseados no nome da função e dados."""
    constraints = {}

    # Constraints para transformações de dados
    if "noise" in func_name.lower():
        constraints.update({"type": "data_noise", "preserve_shape": True, "preserve_bounds": True})

    if "outlier" in func_name.lower():
        constraints.update(
            {
                "type": "outlier_injection",
                "preserve_cardinality": True,
                "outlier_ratio_bounds": [0.0, 0.5],
            }
        )

    if "scaling" in func_name.lower() or "normalization" in func_name.lower():
        constraints.update({"type": "scaling", "preserve_shape": True, "preserve_ordering": True})

    # Constraints para transformações de labels
    if "label" in func_name.lower():
        constraints.update(
            {
                "type": "label_transformation",
                "preserve_class_balance": True,
                "label_ratio_bounds": [0.0, 1.0],
            }
        )

    # Constraints para transformações de parâmetros
    if "param" in func_name.lower():
        constraints.update(
            {
                "type": "parameter_transformation",
                "preserve_validity": True,
                "parameter_bounds_check": True,
            }
        )

    # Adicionar constraints de integridade básicos
    if input_data is not None and output_data is not None:
        try:
            import numpy as np

            if hasattr(input_data, "shape") and hasattr(output_data, "shape"):
                constraints["input_shape"] = list(input_data.shape)
                constraints["output_shape"] = list(output_data.shape)
        except ImportError:
            pass

    return constraints


def verify_model_operation(
    operation_type: str = "training",
    constraints: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
    timeout: float = 60.0,
):
    """
    Decorador para verificação de operações de modelo (train, predict, evaluate).

    Args:
        operation_type: Tipo de operação (training, prediction, evaluation)
        constraints: Constraints específicos
        enabled: Se verificação está habilitada
        timeout: Timeout para verificação
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            if enabled and verification_manager.enabled:
                # Preparar constraints baseados no tipo de operação
                verification_constraints = constraints or {}
                verification_constraints.update(
                    {"operation_type": operation_type, "model_type": self.__class__.__name__}
                )

                # Executar verificação
                verification_result = verification_manager.verify(
                    name=f"{self.__class__.__name__}.{func.__name__}",
                    constraints=verification_constraints,
                    parameters={
                        "operation_type": operation_type,
                        "model_params": getattr(self, "get_params", lambda: {})(),
                        **kwargs,
                    },
                    timeout=timeout,
                )

                logger.info(
                    f"Model verification for {func.__name__}: {verification_result.status.value}"
                )

            return result

        return wrapper

    return decorator


def verify_pipeline_step(
    step_name: str, constraints: Optional[Dict[str, Any]] = None, enabled: bool = True
):
    """
    Decorador para verificação de etapas do pipeline.

    Args:
        step_name: Nome da etapa do pipeline
        constraints: Constraints específicos
        enabled: Se verificação está habilitada
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if enabled and verification_manager.enabled:
                verification_constraints = constraints or {
                    "pipeline_step": step_name,
                    "step_integrity": True,
                }

                verification_result = verification_manager.verify(
                    name=f"pipeline.{step_name}",
                    constraints=verification_constraints,
                    input_data=args[0] if args else None,
                    output_data=result,
                    parameters=kwargs,
                )

                logger.debug(
                    f"Pipeline step verification for {step_name}: {verification_result.status.value}"
                )

            return result

        return wrapper

    return decorator
