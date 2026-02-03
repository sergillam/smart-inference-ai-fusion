"""Unified inference pipeline for data, label, and parameter perturbations."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from smart_inference_ai_fusion.inference.engine.inference_engine import InferenceEngine
from smart_inference_ai_fusion.inference.engine.label_runner import LabelInferenceEngine
from smart_inference_ai_fusion.inference.engine.param_runner import ParameterInferenceEngine
from smart_inference_ai_fusion.utils.solver_comparison import add_comparison_result
from smart_inference_ai_fusion.utils.verification_config import get_verification_config
from smart_inference_ai_fusion.utils.verification_report import report_verification_results
from smart_inference_ai_fusion.verification.core.formal_verification import verification_manager
from smart_inference_ai_fusion.verification.core.plugin_interface import registry
from smart_inference_ai_fusion.verification.decorators import verify_pipeline_step

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

    def __init__(
        self,
        data_noise_config=None,
        label_noise_config=None,
        X_train=None,
        verification_config=None,
    ):
        """Initializes the inference pipeline.

        Args:
            data_noise_config (object, optional):
                Configuration for data noise (features).
            label_noise_config (object, optional):
                Configuration for label noise.
            X_train (Any, optional):
                Training features, may be required by some label perturbations.
            verification_config (dict, optional):
                Configuration for formal verification (deprecated - use env vars).
        """
        self.data_engine = InferenceEngine(data_noise_config) if data_noise_config else None
        self.label_engine = (
            LabelInferenceEngine(label_noise_config, X_train=X_train)
            if label_noise_config
            else None
        )
        self.param_engine = ParameterInferenceEngine()

        # 🎛️ Configuração multi-solver via sistema de configuração global
        self.config = get_verification_config()

        # Configuração de comportamento de erro
        self.fail_on_verification_error = False  # Por padrão, não falha em erros de verificação

        # Compatibilidade com configuração legacy
        if verification_config:
            logger.warning(
                "verification_config parameter is deprecated. Use environment variables instead."
            )

        logger.info(f"🏗️ Pipeline initialized with configuration: {self.config}")

        # Cache de verificadores para performance
        self._verifier_cache = {}
        self._load_verifiers()

    def _load_verifiers(self):
        """Carrega e configura verificadores baseado na configuração."""
        if not self.config.should_verify():
            logger.info("🔹 Formal verification disabled")
            return

        enabled_solvers = self.config.get_enabled_solvers()
        logger.info(f"🔧 Carregando verificadores: {enabled_solvers}")

        for solver_name in enabled_solvers:
            verifier = registry.get_verifier(solver_name)
            if verifier and verifier.is_available():
                self._verifier_cache[solver_name] = verifier
                logger.info(f"✅ Verificador {solver_name} carregado")
            else:
                logger.warning(f"⚠️ Verifier {solver_name} not available")

        if not self._verifier_cache:
            logger.warning("❌ No verifiers available - verification disabled")

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
        if self.config.should_verify():
            verification_results = self._verify_parameters_multi_solver(
                "pre_perturbation", base_params
            )

        perturbed_params = self.param_engine.apply(base_params)

        # Aplicar verificação de parâmetros após a perturbação
        if self.config.should_verify():
            post_verification_results = self._verify_parameters_multi_solver(
                "post_perturbation", perturbed_params, base_params
            )

        model = model_class(**perturbed_params)

        # Verificar integridade do modelo criado
        if self.config.should_verify():
            model_verification_results = self._verify_model_integrity_multi_solver(
                model, perturbed_params
            )

        # Preparar log com resultados de verificação
        log_dict = {"perturbed_params": perturbed_params}
        if self.config.should_verify():
            log_dict.update(
                {
                    "verification_results": {
                        "pre_perturbation": verification_results,
                        "post_perturbation": post_verification_results,
                        "model_integrity": model_verification_results,
                    }
                }
            )

            # Adicionar resultados ao framework de comparação
            if len(self._verifier_cache) > 1:
                add_comparison_result("param_pre_perturbation", verification_results)
                add_comparison_result("param_post_perturbation", post_verification_results)
                add_comparison_result("model_integrity", model_verification_results)

        return model, log_dict

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
        if self.config and self.config.should_verify():
            self._verify_data_integrity("input_data", X_train, X_test)

        X_train_perturbed, X_test_perturbed = self.data_engine.apply(X_train, X_test)

        # Verificar integridade dos dados após perturbação
        if self.config and self.config.should_verify():
            self._verify_data_integrity(
                "output_data",
                X_train_perturbed,
                X_test_perturbed,
                original_train=X_train,
                original_test=X_test,
            )

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
        if self.config and self.config.should_verify():
            self._verify_label_integrity("input_labels", y_train, y_test)

        y_train_perturbed, y_test_perturbed = self.label_engine.apply(
            y_train=y_train, y_test=y_test, model=model, X_train=X_train, X_test=X_test
        )

        # Verificar integridade dos labels após perturbação
        if self.config and self.config.should_verify():
            self._verify_label_integrity(
                "output_labels",
                y_train_perturbed,
                y_test_perturbed,
                original_train=y_train,
                original_test=y_test,
            )

        return y_train_perturbed, y_test_perturbed

    # ======= MÉTODOS DE VERIFICAÇÃO MULTI-SOLVER =======

    def _verify_parameters_multi_solver(
        self,
        step_name: str,
        params: Dict[str, Any],
        original_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Verifica parâmetros usando múltiplos solvers."""
        if not self._verifier_cache:
            return {}

        constraints = self._build_parameter_constraints(params, original_params)
        results = {}

        if self.config.parallel_solvers and len(self._verifier_cache) > 1:
            # Verificação paralela
            results = self._verify_parallel(f"pipeline.parameters.{step_name}", constraints, params)
        else:
            # Verificação sequencial
            for solver_name, verifier in self._verifier_cache.items():
                try:
                    result = self._verify_with_solver(
                        verifier, f"pipeline.parameters.{step_name}", constraints, params
                    )
                    results[solver_name] = result
                except Exception as e:
                    logger.error(f"❌ Verification error with {solver_name}: {e}")
                    results[solver_name] = {"error": str(e), "status": "ERROR"}

        # Análise comparativa se múltiplos solvers
        if len(results) > 1 and self.config.compare_solvers:
            results["comparison"] = self._compare_verification_results(results)

        return results

    def _verify_model_integrity_multi_solver(self, model, params: Dict[str, Any]) -> Dict[str, Any]:
        """Verifica integridade do modelo usando múltiplos solvers."""
        if not self._verifier_cache:
            return {}

        constraints = self._build_model_constraints(model, params)
        results = {}

        for solver_name, verifier in self._verifier_cache.items():
            try:
                result = self._verify_with_solver(
                    verifier,
                    "pipeline.model.integrity",
                    constraints,
                    {"model_params": params, "model_type": type(model).__name__},
                )
                results[solver_name] = result
            except Exception as e:
                logger.error(f"❌ Model verification error with {solver_name}: {e}")
                results[solver_name] = {"error": str(e), "status": "ERROR"}

        return results

    def _verify_parallel(
        self, name: str, constraints: Dict[str, Any], input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executa verificação em paralelo com múltiplos solvers incluindo error handling."""
        from smart_inference_ai_fusion.verification.core.error_handling import should_disable_solver

        results = {}

        # Filtrar solvers ativos (não desabilitados por erros)
        active_verifiers = {
            solver_name: verifier
            for solver_name, verifier in self._verifier_cache.items()
            if not should_disable_solver(solver_name)
        }

        if not active_verifiers:
            logger.warning("❌ No active solvers available for parallel verification")
            return {"error": "No active solvers available"}

        logger.info(f"🔄 Running parallel verification with {len(active_verifiers)} active solvers")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submeter tarefas para cada verificador ativo
            future_to_solver = {
                executor.submit(
                    self._verify_with_solver, verifier, name, constraints, input_data
                ): solver_name
                for solver_name, verifier in active_verifiers.items()
            }

            # Coletar resultados com timeout individual
            for future in as_completed(future_to_solver):
                solver_name = future_to_solver[future]
                try:
                    result = future.result(
                        timeout=self.config.timeout_per_constraint + 10
                    )  # +10s buffer
                    results[solver_name] = result
                    logger.info(f"✅ Verification {solver_name} completed for {name}")
                except TimeoutError:
                    logger.warning(f"⏰ Timeout on verification {solver_name} for {name}")
                    results[solver_name] = {
                        "error": "Verification timeout",
                        "status": "TIMEOUT",
                        "execution_time": self.config.timeout_per_constraint,
                    }
                except Exception as e:
                    logger.error(f"❌ Verification {solver_name} failed: {e}")
                    results[solver_name] = {
                        "error": str(e),
                        "status": "ERROR",
                        "execution_time": 0.0,
                    }

        return results

    def _verify_with_solver(
        self, verifier, name: str, constraints: Dict[str, Any], input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executa verificação com um solver específico incluindo error handling."""
        from smart_inference_ai_fusion.verification.core.error_handling import (
            handle_verification_error,
            should_disable_solver,
        )
        from smart_inference_ai_fusion.verification.core.plugin_interface import VerificationInput

        solver_name = verifier.name

        # Verificar se solver deve ser pulado devido a erros anteriores
        if should_disable_solver(solver_name):
            logger.warning(f"⚠️ Pulando {solver_name} - temporariamente desabilitado")
            return {
                "status": "SKIPPED",
                "message": f"{solver_name} temporarily disabled due to reliability issues",
                "execution_time": 0.0,
                "constraints_checked": [],
                "constraints_satisfied": [],
                "constraints_violated": [],
                "details": {"disabled": True},
            }

        try:
            verification_input = VerificationInput(
                name=name,
                constraints=constraints,
                input_data=input_data,
                timeout=self.config.timeout_per_constraint,
            )

            result = verifier.verify(verification_input)

            return {
                "status": result.status.value,
                "message": result.message,
                "execution_time": result.execution_time,
                "constraints_checked": result.constraints_checked or [],
                "constraints_satisfied": result.constraints_satisfied or [],
                "constraints_violated": result.constraints_violated or [],
                "details": result.details or {},
            }

        except Exception as e:
            # Error handling a nível de pipeline
            error_context = {
                "verification_name": name,
                "constraints": list(constraints.keys()),
                "timeout": self.config.timeout_per_constraint,
                "input_data_keys": list(input_data.keys()),
            }

            error_result = handle_verification_error(
                e, solver_name, "pipeline_verification", error_context
            )

            # Aplicar estratégias de recuperação se disponíveis
            if error_result.get("action") == "switch_solver":
                new_solver_name = error_result.get("new_solver")
                if new_solver_name and new_solver_name in self._verifier_cache:
                    logger.info(f"🔄 Tentando fallback: {solver_name} → {new_solver_name}")
                    fallback_verifier = self._verifier_cache[new_solver_name]
                    try:
                        return self._verify_with_solver(
                            fallback_verifier, name, constraints, input_data
                        )
                    except Exception as fallback_error:
                        logger.error(f"❌ Fallback also failed: {fallback_error}")

            return {
                "status": "ERROR",
                "message": error_result.get(
                    "message", f"{solver_name} verification failed: {str(e)}"
                ),
                "execution_time": 0.0,
                "constraints_checked": [],
                "constraints_satisfied": [],
                "constraints_violated": [],
                "details": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_handling": error_result,
                },
            }

    def _compare_verification_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compara resultados entre diferentes solvers."""
        comparison = {
            "solvers_count": len([k for k in results.keys() if k != "comparison"]),
            "agreement": {},
            "performance": {},
            "status_summary": {},
        }

        solver_results = {k: v for k, v in results.items() if k != "comparison"}

        # Análise de concordância de status
        statuses = [result.get("status", "ERROR") for result in solver_results.values()]
        unique_statuses = set(statuses)

        comparison["agreement"]["status_agreement"] = len(unique_statuses) == 1
        comparison["agreement"]["common_status"] = (
            list(unique_statuses)[0] if len(unique_statuses) == 1 else None
        )

        # Análise de performance
        execution_times = {
            solver: result.get("execution_time", float("inf"))
            for solver, result in solver_results.items()
        }

        if execution_times:
            fastest_solver = min(execution_times, key=execution_times.get)
            comparison["performance"]["fastest_solver"] = fastest_solver
            comparison["performance"]["execution_times"] = execution_times
            comparison["performance"]["time_difference"] = {
                solver: time - execution_times[fastest_solver]
                for solver, time in execution_times.items()
            }

        # Resumo de status por solver
        comparison["status_summary"] = {
            solver: result.get("status", "ERROR") for solver, result in solver_results.items()
        }

        return comparison

    def _build_parameter_constraints(
        self, params: Dict[str, Any], original_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Constrói constraints estruturados para verificação de parâmetros."""
        constraints = {
            "type_safety": True,
            "bounds": {"min": 0.0, "max": 1000.0, "strict": False, "allow_nan": False},
            "range_check": {
                "type": "continuous",
                "valid_ranges": [(0.0, 1.0), (1.0, 100.0)],
                "tolerance": 1e-9,
            },
            "real_arithmetic": True,
        }

        if original_params is not None:
            constraints.update(
                {
                    "bounds": True,
                    "type_safety": True,
                    "parameter_drift": {
                        "max_relative_change": 0.5,  # Máximo 50% de mudança
                        "forbidden_sign_flip": True,
                    },
                }
            )

        return constraints

    def _build_model_constraints(self, model, params: Dict[str, Any]) -> Dict[str, Any]:
        """Constrói constraints para verificação de integridade do modelo."""
        return {
            "model_instantiation": True,
            "parameter_consistency": True,
            "type_safety": True,
            "attribute_check": {
                "required_attributes": ["fit", "predict"],
                "sklearn_compatibility": True,
            },
        }

    def _build_data_constraints(self, original_train=None, original_test=None) -> Dict[str, Any]:
        """Constrói constraints para verificação de dados."""
        constraints = {
            "shape_preservation": True,
            "type_safety": True,
            "bounds": {"min": -1000.0, "max": 1000.0, "strict": False, "allow_nan": False},
            "range_check": {
                "type": "continuous",
                "valid_ranges": [(-100.0, 100.0)],
                "allow_empty": False,
                "tolerance": 1e-6,
            },
        }
        if original_train is not None and original_test is not None:
            constraints.update({"bounds": True, "non_negative": True})
        return constraints

    def _build_label_constraints(self, original_train=None, original_test=None) -> Dict[str, Any]:
        """Constrói constraints para verificação de labels."""
        constraints = {
            "shape_preservation": True,
            "type_safety": True,
            "bounds": {"min": 0, "max": 10, "strict": False, "allow_nan": False},
            "range_check": {
                "type": "discrete",
                "discrete_values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "allow_empty": False,
            },
        }
        if original_train is not None and original_test is not None:
            constraints.update({"bounds": True, "integer_arithmetic": True})
        return constraints

    def _build_input_data(self, original_train, original_test) -> Optional[Dict[str, Any]]:
        """Constrói dados de entrada para verificação."""
        if original_train is not None:
            return {"train": original_train, "test": original_test}
        return None

    def _run_multi_solver_verification(
        self,
        comparison_key: str,
        pipeline_prefix: str,
        constraints: Dict[str, Any],
        input_data: Any,
        output_data: Any,
        model_name: str,
    ):
        """Executa verificação com múltiplos solvers e adiciona à comparação."""
        if self._verifier_cache:
            results = self._verify_generic_multi_solver(
                f"{pipeline_prefix}.{comparison_key.split('_')[-1]}",
                constraints,
                input_data,
                output_data,
                model_name.replace("Pipeline", ""),
            )
            if len(results) > 1:
                add_comparison_result(comparison_key, results)

    def _verify_generic_multi_solver(
        self,
        name: str,
        constraints: Dict[str, Any],
        input_data: Any,
        output_data: Any,
        category: str,
    ) -> Dict[str, Any]:
        """Verificação genérica com múltiplos solvers."""
        results = {}
        for solver_name, verifier in self._verifier_cache.items():
            try:
                result = self._verify_with_solver(
                    verifier, name, constraints, {"input": input_data, "output": output_data}
                )
                results[solver_name] = result
                logger.info(f"✅ {category} verification with {solver_name} completed")
            except Exception as e:
                logger.error(f"❌ {category} verification error with {solver_name}: {e}")
                results[solver_name] = {"error": str(e), "status": "ERROR"}
        return results

    def _verify_data_integrity(
        self, step_name: str, X_train, X_test, original_train=None, original_test=None
    ):
        """Verifica integridade dos dados com múltiplos solvers para comparação justa."""
        constraints = self._build_data_constraints(original_train, original_test)
        input_data = self._build_input_data(original_train, original_test)
        output_data = {"train": X_train, "test": X_test}

        self._run_multi_solver_verification(
            f"data_{step_name}",
            "pipeline.data",
            constraints,
            input_data,
            output_data,
            "DataPipeline",
        )

    def _verify_data_multi_solver(
        self, step_name: str, constraints: Dict[str, Any], input_data: Any, output_data: Any
    ) -> Dict[str, Any]:
        """Verifica dados usando múltiplos solvers para comparação justa."""
        return self._verify_generic_multi_solver(
            f"pipeline.data.{step_name}", constraints, input_data, output_data, "Data"
        )

    def _verify_label_integrity(
        self, step_name: str, y_train, y_test, original_train=None, original_test=None
    ):
        """Verifica integridade dos labels com múltiplos solvers para comparação justa."""
        constraints = self._build_label_constraints(original_train, original_test)
        input_data = self._build_input_data(original_train, original_test)
        output_data = {"train": y_train, "test": y_test}

        self._run_multi_solver_verification(
            f"label_{step_name}",
            "pipeline.labels",
            constraints,
            input_data,
            output_data,
            "LabelPipeline",
        )

    def _verify_label_multi_solver(
        self, step_name: str, constraints: Dict[str, Any], input_data: Any, output_data: Any
    ) -> Dict[str, Any]:
        """Verifica labels usando múltiplos solvers para comparação justa."""
        return self._verify_generic_multi_solver(
            f"pipeline.labels.{step_name}", constraints, input_data, output_data, "Label"
        )

    def _verify_parameters(
        self,
        step_name: str,
        params: Dict[str, Any],
        original_params: Optional[Dict[str, Any]] = None,
    ):
        """Verifica integridade dos parâmetros com constraints estruturados."""
        # Mapear para constraints compatíveis com Z3 com dados estruturados
        constraints = {
            "type_safety": True,  # Compatible with Z3
            "bounds": {"min": 0.0, "max": 1000.0, "strict": False, "allow_nan": False},
            "range_check": {
                "type": "continuous",
                "valid_ranges": [(0.0, 1.0), (1.0, 100.0)],  # Para diferentes tipos de parâmetros
                "tolerance": 1e-9,
            },
            "real_arithmetic": True,  # For numeric parameters
        }

        if original_params is not None:
            constraints.update(
                {
                    "bounds": True,  # Preserve parameter bounds
                    "type_safety": True,  # Maintain parameter types
                }
            )

        verification_result = verification_manager.verify(
            name=f"pipeline.parameters.{step_name}",
            constraints=constraints,
            input_data=original_params,
            output_data=params,
            timeout=self.config.timeout_per_constraint if self.config else 30.0,
        )

        # Reportar resultados detalhados
        if verification_result:
            report_verification_results(
                verification_result=verification_result,
                model_name="ParameterPipeline",
                dataset_name="unknown",
                transformation_name=f"parameter_integrity_{step_name}",
            )

        if self.fail_on_verification_error and verification_result.status.value == "FAILED":
            raise RuntimeError(
                f"Parameter verification failed at {step_name}: {verification_result.message}"
            )

    def _verify_model_integrity(self, model, params: Dict[str, Any]):
        """Verifica integridade do modelo criado."""
        # Mapear para constraints compatíveis com Z3
        constraints = {
            "type_safety": True,  # Compatible with Z3
            "bounds": True,  # Compatible with Z3
            "range_check": True,  # Compatible with Z3
        }

        verification_result = verification_manager.verify(
            name=f"pipeline.model.{model.__class__.__name__}",
            constraints=constraints,
            input_data=params,
            output_data=model,
            timeout=self.config.timeout_per_constraint if self.config else 30.0,
        )

        if self.fail_on_verification_error and verification_result.status.value == "FAILED":
            raise RuntimeError(f"Model verification failed: {verification_result.message}")
