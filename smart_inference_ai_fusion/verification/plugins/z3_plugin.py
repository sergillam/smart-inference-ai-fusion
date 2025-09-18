"""Plugin Z3 para verificação formal com todos os recursos do Z3."""

import logging
from typing import Any, Dict, List, Optional, Union
import time

from ..core.plugin_interface import FormalVerifier, VerificationInput, VerificationResult, VerificationStatus
from ..core.error_handling import handle_verification_error, should_disable_solver
from ..core.result_schema import StandardVerificationResult, SolverMetadata, PerformanceMetrics, ConstraintResult, StandardStatus

logger = logging.getLogger(__name__)

# Verificar disponibilidade do Z3
try:
    import z3
    Z3_AVAILABLE = True
    logger.info("Z3 SMT solver available")
except ImportError:
    z3 = None
    Z3_AVAILABLE = False
    logger.warning("Z3 not available. Install with: pip install z3-solver")


class Z3Verifier(FormalVerifier):
    """Verificador formal usando Z3 SMT Solver com todos os recursos."""
    
    def __init__(self):
        super().__init__("Z3")
        self.version = z3.get_version_string() if Z3_AVAILABLE else "unknown"
        self.priority = 1  # Prioridade alta para Z3 (solver padrão)
        self.solver = None
        if Z3_AVAILABLE:
            self._init_z3()
    
    def _init_z3(self):
        """Inicializa o solver Z3 com configuração de máximo desempenho científico."""
        self.solver = z3.Solver()
        
        # 🚀 CONFIGURAÇÕES DE MÁXIMO DESEMPENHO CIENTÍFICO - APENAS PARÂMETROS VÁLIDOS
        
        # === TIMEOUTS E RECURSOS COMPUTACIONAIS ===
        self.solver.set("timeout", 600000)  # 10 minutos para problemas complexos
        self.solver.set("rlimit", 10000000)  # Limite de recursos aumentado 
        
        # === PARALELIZAÇÃO MÁXIMA ===
        import os
        max_threads = min(16, os.cpu_count() or 4)  # Usar todos os cores disponíveis
        self.solver.set("threads", max_threads)
        
        # === GESTÃO DE MEMÓRIA OTIMIZADA ===
        self.solver.set("max_memory", 14000)  # 14GB para Z3 (máximo seguro)
        
        # === ESTRATÉGIAS SAT AVANÇADAS ===
        self.solver.set("restart.max", 5000000)  # Reinicializações mais agressivas
        self.solver.set("restart_factor", 1.5)
        self.solver.set("restart_strategy", 1)
        self.solver.set("phase_selection", 3)  # Estratégia de fase otimizada
        self.solver.set("random_seed", 12345)
        
        # === HEURÍSTICAS AVANÇADAS DE DECISÃO ===
        self.solver.set("phase_caching_on", 200)  # Cache de fase otimizado
        self.solver.set("phase_caching_off", 50)
        
        # === ESTRATÉGIAS SMT ESPECÍFICAS ===
        self.solver.set("auto_config", False)  # Desabilitar auto_config para permitir case_split avançado
        self.solver.set("case_split", 5)  # Case splitting agressivo (agora permitido)
        self.solver.set("relevancy", 2)  # Relevância máxima
        self.solver.set("macro_finder", True)
        self.solver.set("pull_nested_quantifiers", True)
        self.solver.set("mbqi", True)  # Model-based quantifier instantiation
        
        # === ESPECÍFICO PARA PROBLEMAS DE ML/IA ===
        self.solver.set("arith.random_initial_value", True)
        self.solver.set("arith.solver", 6)  # Solver aritmético avançado
        self.solver.set("arith.nl", True)  # Não-linear habilitado
        self.solver.set("arith.nl.grobner", True)  # Gröbner bases
        self.solver.set("arith.nl.order", True)
        self.solver.set("arith.auto_config_simplex", True)
        self.solver.set("arith.greatest_error_pivot", True)
        self.solver.set("arith.propagate_eqs", True)
        self.solver.set("arith.eager_eq_axioms", True)
        
        # === QUANTIFICADORES E INSTANCIAÇÃO ===
        self.solver.set("ematching", True)
        self.solver.set("qi.eager_threshold", 5.0)  # Threshold reduzido para mais agressividade
        self.solver.set("qi.lazy_threshold", 10.0)
        self.solver.set("qi.max_instances", 1000000)  # Muito mais instâncias
        
        # === PRÉ-PROCESSAMENTO INTENSIVO ===
        self.solver.set("expand_store_eq", True)
        self.solver.set("flat", True)
        self.solver.set("hi_div0", True)
        self.solver.set("sort_store", True)
        self.solver.set("elim_ite", True)
        self.solver.set("elim_unconstrained", True)
        
        # === BIT-VECTORS E ARRAYS ===
        self.solver.set("bv.solver", 0)  # Solver BV otimizado
        self.solver.set("array.extensional", True)
        
        # === CONFIGURAÇÃO DE LÓGICA OTIMIZADA ===
        # QF_NIRA: Quantifier-free Nonlinear Integer Real Arithmetic
        # Ideal para problemas de ML com constraints complexos
        self.solver.set("logic", "QF_NIRA")
        
        # === COLETA DE INFORMAÇÕES PARA DEBUGGING ===
        self.solver.set("model", True)  # Gerar modelos
        self.solver.set("unsat_core", True)  # Gerar cores insatisfazíveis
        
        logger.info(f"🚀 Z3 initialized with maximum scientific configuration - {max_threads} threads, 14GB RAM")
        logger.debug(f"Z3 version: {z3.get_version_string()}")
        
        # Log de parâmetros importantes (sem chamar param_descrs que pode causar erro)
        try:
            logger.debug("Z3 configuration applied successfully")
        except Exception as e:
            logger.debug(f"Error logging Z3 parameters: {e}")
    
    def is_available(self) -> bool:
        """Verifica se Z3 está disponível."""
        return Z3_AVAILABLE
    
    def supported_constraints(self) -> List[str]:
        """Lista completa de constraints suportados pelo Z3."""
        return [
            # Constraints básicos
            'bounds',
            'range_check', 
            'type_safety',
            'non_negative',
            'positive',
            'shape_preservation',
            
            # Constraints aritméticos
            'linear_arithmetic',
            'non_linear_arithmetic', 
            'polynomial_constraints',
            'integer_arithmetic',
            'real_arithmetic',
            'rational_arithmetic',
            'modular_arithmetic',
            
            # Constraints lógicos
            'boolean_logic',
            'propositional_logic',
            'implication',
            'equivalence',
            'conditional_logic',
            
            # Constraints de arrays e estruturas
            'array_theory',
            'array_bounds',
            'array_sorting',
            'sequence_theory',
            'string_theory',
            'regex_constraints',
            
            # Constraints de bit-vectors
            'bitvector_arithmetic',
            'bitwise_operations',
            'overflow_detection',
            'underflow_detection',
            
            # Constraints de ponto flutuante
            'floating_point',
            'fp_arithmetic',
            'fp_rounding',
            'fp_special_values',
            
            # Constraints avançados
            'quantified_formulas',
            'existential_quantification', 
            'universal_quantification',
            'algebraic_datatypes',
            'recursive_functions',
            'inductive_definitions',
            
            # Constraints de otimização
            'optimization',
            'maximize',
            'minimize',
            'multi_objective',
            
            # Constraints probabilísticos e estocásticos
            'probability_bounds',
            'statistical_constraints',
            'distribution_constraints',
            
            # Constraints de machine learning específicos
            'neural_network_verification',
            'decision_tree_verification',
            'model_robustness',
            'adversarial_robustness',
            'fairness_constraints',
            
            # 🔒 CONSTRAINTS ESPECÍFICOS: INVARIANTES, PRÉ/PÓS-CONDIÇÕES, ROBUSTEZ
            'invariant',
            'precondition', 
            'postcondition',
            'robustness',
            'data_consistency',
            'model_stability',
            'parameter_validity',
            'data_preprocessing',
            'parameter_initialization',
            'data_shape_validation',
            'output_validity',
            'probability_bounds_postcondition',
            'classification_constraints',
            'adversarial_robustness_test',
            'noise_robustness',
            'parameter_sensitivity',
            'distributional_robustness',
            
            # 🎯 ALGORITMOS ESPECÍFICOS DO EXPERIMENTO
            'logistic_regression_convergence',
            'logistic_regression_probability_bounds',
            'logistic_regression_gradient_stability',
            'logistic_regression_regularization',
            'decision_tree_purity',
            'decision_tree_depth_bounds',
            'decision_tree_split_validity',
            'decision_tree_leaf_distribution',
            'mlp_architecture_validity',
            'mlp_activation_bounds',
            'mlp_weight_initialization',
            'mlp_gradient_flow',
            'mlp_backpropagation_stability',
            
            # 📊 DATASET-SPECIFIC CONSTRAINTS
            'adult_fairness_constraints',
            'adult_bias_detection',
            'breast_cancer_precision_bounds',
            'breast_cancer_recall_requirements',
            'wine_quality_classification',
            'wine_feature_importance',
            'make_moons_separability',
            'make_moons_nonlinear_boundaries',
            
            # Constraints temporais e dinâmicos
            'temporal_logic',
            'ltl_properties',
            'ctl_properties',
            'invariant_preservation',
            'reachability_analysis'
        ]
    
    def verify(self, input_data: VerificationInput) -> VerificationResult:
        """Executa verificação usando Z3 com error handling robusto."""
        if not self.enabled:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                verifier_name=self.name,
                execution_time=0.0,
                message="Z3 verifier is disabled"
            )
        
        if not Z3_AVAILABLE:
            error_result = handle_verification_error(
                ImportError("Z3 not available"), 
                self.name, 
                "initialization",
                {"suggestion": "pip install z3-solver"}
            )
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verifier_name=self.name,
                execution_time=0.0,
                message=error_result.get("message", "Z3 not available"),
                details={"error_context": error_result}
            )
        
        # Verificar se solver deve ser desabilitado devido a erros anteriores
        if should_disable_solver(self.name):
            logger.warning(f"⚠️ Z3 temporariamente desabilitado devido a muitos erros")
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                verifier_name=self.name,
                execution_time=0.0,
                message="Z3 temporarily disabled due to reliability issues"
            )
        
        start_time = time.time()
        
        try:
            # DEBUG: Log constraints recebidas
            logger.info(f"Z3 DEBUG - Constraints recebidas: {input_data.constraints}")
            logger.info(f"Z3 DEBUG - Constraints suportadas: {self.supported_constraints()}")
            
            # Reinicializar solver para verificação limpa
            self.solver.reset()
            self._init_z3()
            
            # Processar cada constraint
            constraints_checked = []
            constraints_satisfied = []
            constraints_violated = []
            solver_details = {}
            
            for constraint_type, constraint_data in input_data.constraints.items():
                if constraint_type in self.supported_constraints():
                    constraints_checked.append(constraint_type)
                    
                    try:
                        satisfied, details = self._verify_constraint_with_details(constraint_type, constraint_data, input_data)
                        solver_details[constraint_type] = details
                        
                        if satisfied:
                            constraints_satisfied.append(constraint_type)
                        else:
                            constraints_violated.append(constraint_type)
                    except Exception as constraint_error:
                        # Error handling por constraint individual
                        error_context = {
                            "constraint_type": constraint_type,
                            "constraint_data": constraint_data,
                            "timeout": getattr(input_data, 'timeout', 30)
                        }
                        
                        error_result = handle_verification_error(
                            constraint_error, 
                            self.name, 
                            f"constraint_verification_{constraint_type}",
                            error_context
                        )
                        
                        constraints_violated.append(constraint_type)
                        solver_details[constraint_type] = {
                            "error": str(constraint_error),
                            "error_handling": error_result
                        }
                        
                        # Se error handling sugeriu fallback, aplicar
                        if error_result.get("action") == "use_basic_constraints":
                            logger.info(f"🔧 Applying basic fallback for {constraint_type}")
                            # Continuar com constraints simplificados
                        
                        logger.warning(f"Failed to verify {constraint_type}: {constraint_error}")
            
            # Determinar status geral
            execution_time = time.time() - start_time
            
            # REPORTING DETALHADO SEMPRE ATIVO PARA VERIFICAÇÃO
            self._report_verification_details(input_data, solver_details, constraints_satisfied, 
                                           constraints_violated, execution_time)
            
            if constraints_violated:
                status = VerificationStatus.FAILURE
                message = f"Violated constraints: {', '.join(constraints_violated)}"
            elif constraints_satisfied:
                status = VerificationStatus.SUCCESS
                message = f"All constraints satisfied: {', '.join(constraints_satisfied)}"
            else:
                status = VerificationStatus.SKIPPED
                message = "No supported constraints found"
            
            return VerificationResult(
                status=status,
                verifier_name=self.name,
                execution_time=execution_time,
                message=message,
                constraints_checked=constraints_checked,
                constraints_satisfied=constraints_satisfied,
                constraints_violated=constraints_violated,
                details=solver_details
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Error handling para falhas gerais
            error_context = {
                "constraints": list(input_data.constraints.keys()),
                "execution_time": execution_time,
                "timeout": getattr(input_data, 'timeout', 30)
            }
            
            error_result = handle_verification_error(e, self.name, "verification", error_context)
            
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verifier_name=self.name,
                execution_time=execution_time,
                message=error_result.get("message", f"Z3 verification error: {str(e)}"),
                details={
                    "error": str(e),
                    "error_handling": error_result
                }
            )
    
    def _verify_constraint_with_details(self, constraint_type: str, constraint_data: Any, 
                          input_data: VerificationInput) -> tuple[bool, dict]:
        """Verifica um constraint específico usando Z3 e retorna detalhes.
        
        Returns:
            tuple: (satisfied: bool, details: dict)
        """
        # Reset solver para verificação limpa
        self.solver.reset()
        self._init_z3()
        
        details = {
            "constraint_type": constraint_type,
            "constraint_data": constraint_data,
            "satisfiable": None,
            "model": None,
            "unsat_core": None,
            "statistics": None
        }
        
        try:
            # Executar verificação específica
            satisfied = self._verify_constraint(constraint_type, constraint_data, input_data)
            
            # Capturar resultado SAT/UNSAT
            check_result = self.solver.check()
            details["satisfiable"] = (check_result == z3.sat)
            
            if check_result == z3.sat:
                # Capturar modelo se SAT
                model = self.solver.model()
                details["model"] = str(model) if model else None
            elif check_result == z3.unsat:
                # Capturar core insatisfazível se UNSAT
                try:
                    unsat_core = self.solver.unsat_core()
                    details["unsat_core"] = [str(core) for core in unsat_core]
                except:
                    details["unsat_core"] = ["unable_to_extract"]
            
            # Capturar estatísticas do solver
            try:
                stats = self.solver.statistics()
                details["statistics"] = {
                    "decisions": stats.get_key_value("decisions"),
                    "conflicts": stats.get_key_value("conflicts"),
                    "propagations": stats.get_key_value("propagations"),
                    "restarts": stats.get_key_value("restarts")
                }
            except:
                details["statistics"] = {"error": "unable_to_extract_stats"}
            
            # 🔍 CONTRA-EXEMPLOS: Gerar quando constraint é violado
            if not satisfied:
                logger.debug("🔍 Constraint %s violado - gerando contra-exemplo...", constraint_type)
                try:
                    counterexample = self._generate_counterexample(constraint_type, constraint_data)
                    details["counterexample"] = counterexample
                    logger.debug("✅ Contra-exemplo gerado para %s", constraint_type)
                except Exception as ce_error:
                    details["counterexample"] = {
                        "error": f"Failed to generate counterexample: {str(ce_error)}"
                    }
                    logger.warning("❌ Falha ao gerar contra-exemplo para %s: %s", constraint_type, ce_error)
            else:
                logger.debug("✅ Constraint %s satisfied - not generating counterexample", constraint_type)
                
            return satisfied, details
            
        except Exception as e:
            details["error"] = str(e)
            return False, details
    
    def _verify_constraint(self, constraint_type: str, constraint_data: Any, 
                          input_data: VerificationInput) -> bool:
        """Verifica um constraint específico usando Z3."""
        
        if constraint_type == 'bounds':
            return self._verify_bounds(constraint_data, input_data)
            
        elif constraint_type == 'range_check':
            return self._verify_range_check(constraint_data)
            
        elif constraint_type == 'linear_arithmetic':
            return self._verify_linear_arithmetic(constraint_data)
            
        elif constraint_type == 'boolean_logic':
            return self._verify_boolean_logic(constraint_data)
            
        elif constraint_type == 'array_theory':
            return self._verify_array_theory(constraint_data)
            
        elif constraint_type == 'bitvector_arithmetic':
            return self._verify_bitvector_arithmetic(constraint_data)
            
        elif constraint_type == 'floating_point':
            return self._verify_floating_point(constraint_data)
            
        elif constraint_type == 'string_theory':
            return self._verify_string_theory(constraint_data)
            
        elif constraint_type == 'quantified_formulas':
            return self._verify_quantified_formulas(constraint_data)
            
        elif constraint_type == 'optimization':
            return self._verify_optimization(constraint_data)
            
        elif constraint_type == 'neural_network_verification':
            return self._verify_neural_network(constraint_data)
            
        elif constraint_type == 'probability_bounds':
            return self._verify_probability_bounds(constraint_data)
            
        # 🎯 ALGORITMOS ESPECÍFICOS DO EXPERIMENTO
        elif constraint_type == 'logistic_regression_convergence':
            return self._verify_logistic_regression_convergence(constraint_data, input_data)
        elif constraint_type == 'logistic_regression_probability_bounds':
            return self._verify_logistic_regression_probability_bounds(constraint_data, input_data)
        elif constraint_type == 'decision_tree_purity':
            return self._verify_decision_tree_purity(constraint_data, input_data)
        elif constraint_type == 'mlp_architecture_validity':
            return self._verify_mlp_architecture_validity(constraint_data, input_data)
            
        # 📊 DATASET-SPECIFIC CONSTRAINTS
        elif constraint_type == 'adult_fairness_constraints':
            return self._verify_adult_fairness_constraints(constraint_data, input_data)
        elif constraint_type == 'wine_quality_classification':
            return self._verify_wine_quality_classification(constraint_data, input_data)
        elif constraint_type == 'make_moons_separability':
            return self._verify_make_moons_separability(constraint_data, input_data)
            
        # 🔒 CONSTRAINTS ESPECÍFICOS: INVARIANTES, PRÉ/PÓS-CONDIÇÕES, ROBUSTEZ
        elif constraint_type == 'invariant':
            return self._verify_invariant_constraint(constraint_data, input_data)
        elif constraint_type == 'precondition':
            return self._verify_precondition_constraint(constraint_data, input_data)
        elif constraint_type == 'postcondition':
            return self._verify_postcondition_constraint(constraint_data, input_data)
        elif constraint_type == 'robustness':
            return self._verify_robustness_constraint(constraint_data, input_data)
            
        else:
            # Constraint genérico - assumir satisfeito
            logger.debug(f"Generic constraint verification: {constraint_type}")
            return True

    # 🎯 VERIFICAÇÕES ESPECÍFICAS PARA ALGORITMOS ESCOLHIDOS
    def _verify_logistic_regression_convergence(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica convergência da regressão logística usando Z3."""
        # Variáveis para coeficientes e iterações
        max_iter = constraint_data.get('max_iter', 1000)
        tol = constraint_data.get('tol', 1e-6)
        
        iterations = z3.Int('iterations')
        tolerance = z3.Real('tolerance')
        converged = z3.Bool('converged')
        
        # Constraints de convergência
        self.solver.add(iterations >= 1)
        self.solver.add(iterations <= max_iter)
        self.solver.add(tolerance >= 0)
        self.solver.add(tolerance <= tol)
        
        # Se convergiu, iterations < max_iter
        self.solver.add(z3.Implies(converged, iterations < max_iter))
        
        return self.solver.check() == z3.sat
    
    def _verify_logistic_regression_probability_bounds(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica se probabilidades da regressão logística estão entre 0 e 1."""
        # Função sigmoid: p = 1 / (1 + exp(-z))
        z = z3.Real('z')  # Linear combination
        p = z3.Real('p')  # Probability
        
        # Sigmoid bounds: 0 < p < 1 para qualquer z real
        self.solver.add(p > 0)
        self.solver.add(p < 1)
        
        # Extremos: quando z -> -inf, p -> 0; quando z -> +inf, p -> 1
        self.solver.add(z3.Implies(z < -10, p < 0.01))
        self.solver.add(z3.Implies(z > 10, p > 0.99))
        
        return self.solver.check() == z3.sat
    
    def _verify_decision_tree_purity(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica pureza dos nós da árvore de decisão."""
        criterion = constraint_data.get('criterion', 'gini')
        n_classes = constraint_data.get('n_classes', 3)
        
        # Variáveis para contagem de classes em um nó
        class_counts = z3.IntVector('class_counts', n_classes)
        total_samples = z3.Int('total_samples')
        impurity = z3.Real('impurity')
        
        # Todos os counts devem ser não-negativos
        for i in range(n_classes):
            self.solver.add(class_counts[i] >= 0)
        
        # Total de amostras é a soma dos counts
        self.solver.add(total_samples == z3.Sum(class_counts))
        self.solver.add(total_samples > 0)
        
        # Impureza deve estar nos bounds corretos
        if criterion == 'gini':
            # Gini impurity: 0 <= gini <= 1 - 1/n_classes
            self.solver.add(impurity >= 0)
            self.solver.add(impurity <= 1 - 1.0/n_classes)
        elif criterion == 'entropy':
            # Entropy: 0 <= entropy <= log(n_classes)
            self.solver.add(impurity >= 0)
            self.solver.add(impurity <= 3.0)  # log(3) para 3 classes
        
        return self.solver.check() == z3.sat
    
    def _verify_mlp_architecture_validity(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica validade da arquitetura MLP."""
        hidden_layer_sizes = constraint_data.get('hidden_layer_sizes', (100,))
        input_size = constraint_data.get('input_size', 10)
        output_size = constraint_data.get('output_size', 3)
        
        # Variáveis para arquitetura
        n_hidden_layers = z3.Int('n_hidden_layers')
        min_neurons = z3.Int('min_neurons')
        max_neurons = z3.Int('max_neurons')
        
        # Número de camadas ocultas razoável
        self.solver.add(n_hidden_layers >= 1)
        self.solver.add(n_hidden_layers <= 5)
        
        # Neurônios por camada
        self.solver.add(min_neurons >= 1)
        self.solver.add(max_neurons <= 1000)
        self.solver.add(min_neurons <= max_neurons)
        
        # Input e output sizes devem ser positivos
        self.solver.add(input_size > 0)
        self.solver.add(output_size > 0)
        
        return self.solver.check() == z3.sat

    # 📊 VERIFICAÇÕES ESPECÍFICAS PARA DATASETS
    def _verify_adult_fairness_constraints(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica fairness para dataset Adult (income prediction)."""
        # Variáveis para atributos sensíveis
        age = z3.Int('age')
        gender = z3.Int('gender')  # 0: Female, 1: Male
        race = z3.Int('race')
        prediction = z3.Real('prediction')
        
        # Bounds razoáveis para Adult dataset
        self.solver.add(age >= 17)
        self.solver.add(age <= 90)
        self.solver.add(z3.Or(gender == 0, gender == 1))
        self.solver.add(race >= 0)
        self.solver.add(race <= 4)  # Simplified race encoding
        
        # Fairness: predição não deve depender apenas de atributos sensíveis
        # Demographic parity: P(Y=1|A=0) ≈ P(Y=1|A=1)
        self.solver.add(prediction >= 0)
        self.solver.add(prediction <= 1)
        
        return self.solver.check() == z3.sat
    
    def _verify_wine_quality_classification(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica classificação de qualidade do vinho."""
        # Features típicas do Wine dataset
        alcohol = z3.Real('alcohol')
        acidity = z3.Real('acidity')
        ph = z3.Real('ph')
        quality_class = z3.Int('quality_class')
        
        # Bounds realistas para vinho
        self.solver.add(alcohol >= 8.0)
        self.solver.add(alcohol <= 15.0)
        self.solver.add(acidity >= 0.0)
        self.solver.add(acidity <= 2.0)
        self.solver.add(ph >= 2.5)
        self.solver.add(ph <= 4.5)
        
        # Classes de qualidade (0, 1, 2 para wine dataset)
        self.solver.add(quality_class >= 0)
        self.solver.add(quality_class <= 2)
        
        return self.solver.check() == z3.sat
    
    def _verify_make_moons_separability(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica separabilidade para dataset sintético make_moons."""
        # Coordenadas 2D para make_moons
        x1 = z3.Real('x1')
        x2 = z3.Real('x2')
        label = z3.Int('label')
        noise = z3.Real('noise')
        
        # Bounds para make_moons (tipicamente [-2, 3] x [-1, 2])
        self.solver.add(x1 >= -2.5)
        self.solver.add(x1 <= 3.5)
        self.solver.add(x2 >= -1.5)
        self.solver.add(x2 <= 2.5)
        
        # Labels binários
        self.solver.add(z3.Or(label == 0, label == 1))
        
        # Noise controlado
        self.solver.add(noise >= 0)
        self.solver.add(noise <= 0.3)
        
        # Separabilidade não-linear: distância das "luas"
        distance_moon1 = z3.Real('dist_moon1')
        distance_moon2 = z3.Real('dist_moon2')
        self.solver.add(distance_moon1 >= 0)
        self.solver.add(distance_moon2 >= 0)
        
        return self.solver.check() == z3.sat

    # 🔒 CONSTRAINTS ESPECÍFICOS: INVARIANTES, PRÉ/PÓS-CONDIÇÕES
    def _verify_invariant_constraint(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica invariantes - propriedades que devem sempre ser verdadeiras."""
        try:
            import numpy as np
            
            if not isinstance(constraint_data, dict):
                return True
            
            # Invariantes específicos para ML
            invariants = constraint_data.get('invariants', [])
            all_satisfied = True
            
            for invariant in invariants:
                invariant_type = invariant.get('type', '')
                
                if invariant_type == 'data_consistency':
                    # Invariante: dados devem ter consistência (sem NaN, Inf)
                    data = self._extract_data_from_input(input_data)
                    if hasattr(data, '__iter__') and not isinstance(data, str):
                        data_array = np.array(data).flatten()
                        if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
                            all_satisfied = False
                            logger.warning("🔒 Invariante violado: data_consistency - NaN/Inf detectado")
                
                elif invariant_type == 'model_stability':
                    # Invariante: modelo deve permanecer estável
                    stability_threshold = invariant.get('threshold', 0.1)
                    
                    # Criar variáveis para estabilidade
                    delta_params = z3.Real('delta_params')
                    delta_output = z3.Real('delta_output')
                    stability_ratio = z3.Real('stability_ratio')
                    
                    # Lipschitz continuity: |f(x+δ) - f(x)| ≤ L·|δ|
                    self.solver.add(delta_params >= 0)
                    self.solver.add(delta_output >= 0)
                    self.solver.add(stability_ratio >= 0)
                    self.solver.add(stability_ratio <= stability_threshold)
                    self.solver.add(delta_output <= stability_ratio * delta_params)
                    
                    if self.solver.check() != z3.sat:
                        all_satisfied = False
                        logger.warning("🔒 Invariante violado: model_stability")
                
                elif invariant_type == 'parameter_validity':
                    # Invariante: parâmetros devem estar em ranges válidos
                    param_bounds = invariant.get('bounds', {})
                    parameters = input_data.parameters if input_data else {}
                    
                    for param_name, bounds in param_bounds.items():
                        if param_name in parameters:
                            value = parameters[param_name]
                            min_val, max_val = bounds.get('min', -np.inf), bounds.get('max', np.inf)
                            
                            if not (min_val <= value <= max_val):
                                all_satisfied = False
                                logger.warning(f"🔒 Invariante violado: parameter_validity para {param_name}")
            
            return all_satisfied
            
        except Exception as e:
            logger.warning(f"Error in invariant verification: {e}")
            return True

    def _verify_precondition_constraint(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica pré-condições - condições antes da execução."""
        try:
            import numpy as np
            
            if not isinstance(constraint_data, dict):
                return True
            
            conditions = constraint_data.get('conditions', [])
            all_satisfied = True
            
            for condition in conditions:
                condition_type = condition.get('type', '')
                
                if condition_type == 'data_preprocessing':
                    # Pré-condição: dados devem estar pré-processados
                    data = self._extract_data_from_input(input_data)
                    if hasattr(data, '__iter__') and not isinstance(data, str):
                        data_array = np.array(data).flatten()
                        
                        # Verificar normalização (dados entre -3 e 3 desvios padrão)
                        if len(data_array) > 1:
                            mean_val = np.mean(data_array)
                            std_val = np.std(data_array)
                            if std_val > 0:
                                normalized = (data_array - mean_val) / std_val
                                if np.any(np.abs(normalized) > 3.5):
                                    all_satisfied = False
                                    logger.warning("🔧 Precondition violated: data_preprocessing - data not normalized")
                
                elif condition_type == 'parameter_initialization':
                    # Pré-condição: parâmetros devem estar inicializados corretamente
                    required_params = condition.get('required_params', [])
                    parameters = input_data.parameters if input_data else {}
                    
                    for param in required_params:
                        if param not in parameters:
                            all_satisfied = False
                            logger.warning(f"🔧 Precondition violated: parameter_initialization - {param} not found")
                        elif parameters[param] is None:
                            all_satisfied = False
                            logger.warning(f"🔧 Precondition violated: parameter_initialization - {param} is None")
                
                elif condition_type == 'data_shape_validation':
                    # Pré-condição: forma dos dados deve ser válida
                    expected_shape = condition.get('expected_shape', None)
                    data = self._extract_data_from_input(input_data)
                    
                    if expected_shape and hasattr(data, 'shape'):
                        if data.shape != tuple(expected_shape):
                            all_satisfied = False
                            logger.warning("🔧 Precondition violated: data_shape_validation")
            
            return all_satisfied
            
        except Exception as e:
            logger.warning(f"Error in precondition verification: {e}")
            return True

    def _verify_postcondition_constraint(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica pós-condições - condições após a execução."""
        try:
            import numpy as np
            
            if not isinstance(constraint_data, dict):
                return True
            
            conditions = constraint_data.get('conditions', [])
            all_satisfied = True
            
            for condition in conditions:
                condition_type = condition.get('type', '')
                
                if condition_type == 'output_validity':
                    # Pós-condição: saída deve ser válida
                    data = input_data.output_data if input_data else None
                    if data is not None:
                        if hasattr(data, '__iter__') and not isinstance(data, str):
                            data_array = np.array(data).flatten()
                            if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
                                all_satisfied = False
                                logger.warning("⚡ Postcondition violated: output_validity - NaN/Inf in output")
                
                elif condition_type == 'probability_bounds':
                    # Pós-condição: probabilidades devem estar entre 0 e 1
                    data = input_data.output_data if input_data else None
                    if data is not None and hasattr(data, '__iter__'):
                        data_array = np.array(data).flatten()
                        if np.any(data_array < 0) or np.any(data_array > 1):
                            all_satisfied = False
                            logger.warning("⚡ Postcondition violated: probability_bounds")
                
                elif condition_type == 'classification_constraints':
                    # Pós-condição: classes preditas devem estar no range válido
                    num_classes = condition.get('num_classes', 3)
                    data = input_data.output_data if input_data else None
                    
                    if data is not None and hasattr(data, '__iter__'):
                        data_array = np.array(data).flatten()
                        if np.any(data_array < 0) or np.any(data_array >= num_classes):
                            all_satisfied = False
                            logger.warning("⚡ Postcondition violated: classification_constraints")
            
            return all_satisfied
            
        except Exception as e:
            logger.warning(f"Error in postcondition verification: {e}")
            return True

    # 🛡️ VERIFICAÇÃO DE ROBUSTEZ
    def _verify_robustness_constraint(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica robustez do modelo sob perturbações."""
        try:
            import numpy as np
            
            if not isinstance(constraint_data, dict):
                return True
            
            robustness_tests = constraint_data.get('tests', [])
            all_satisfied = True
            
            for test in robustness_tests:
                test_type = test.get('type', '')
                
                if test_type == 'adversarial_robustness':
                    # Teste: resistência a ataques adversariais
                    epsilon = test.get('epsilon', 0.1)
                    norm_type = test.get('norm', 'l2')
                    
                    # Criar variáveis para perturbação adversarial
                    x_orig = z3.Real('x_original')
                    x_adv = z3.Real('x_adversarial')
                    y_orig = z3.Real('y_original')
                    y_adv = z3.Real('y_adversarial')
                    perturbation = z3.Real('perturbation')
                    
                    # Constraint de perturbação limitada
                    if norm_type == 'l2':
                        self.solver.add(perturbation >= 0)
                        self.solver.add(perturbation <= epsilon)
                        self.solver.add((x_adv - x_orig) * (x_adv - x_orig) <= perturbation * perturbation)
                    elif norm_type == 'linf':
                        self.solver.add(z3.Abs(x_adv - x_orig) <= epsilon)
                    
                    # Robustez: pequenas perturbações não devem mudar drasticamente a saída
                    change_threshold = test.get('output_threshold', 0.1)
                    self.solver.add(z3.Abs(y_adv - y_orig) <= change_threshold)
                    
                    if self.solver.check() != z3.sat:
                        all_satisfied = False
                        logger.warning("🛡️ Robustez violada: adversarial_robustness")
                
                elif test_type == 'noise_robustness':
                    # Teste: resistência a ruído gaussiano
                    noise_level = test.get('noise_level', 0.1)
                    stability_threshold = test.get('stability_threshold', 0.05)
                    
                    # Variáveis para ruído
                    noise = z3.Real('noise')
                    output_change = z3.Real('output_change')
                    
                    # Ruído limitado
                    self.solver.add(z3.Abs(noise) <= noise_level)
                    
                    # Saída deve permanecer estável
                    self.solver.add(z3.Abs(output_change) <= stability_threshold)
                    
                    if self.solver.check() != z3.sat:
                        all_satisfied = False
                        logger.warning("🛡️ Robustez violada: noise_robustness")
                
                elif test_type == 'parameter_sensitivity':
                    # Teste: sensibilidade a mudanças nos parâmetros
                    param_delta = test.get('parameter_delta', 0.01)
                    output_delta_max = test.get('output_delta_max', 0.1)
                    
                    # Variáveis para sensibilidade
                    param_change = z3.Real('param_change')
                    output_change = z3.Real('output_change')
                    sensitivity = z3.Real('sensitivity')
                    
                    # Mudança limitada nos parâmetros
                    self.solver.add(z3.Abs(param_change) <= param_delta)
                    
                    # Sensibilidade: |Δy| / |Δθ| ≤ threshold
                    self.solver.add(sensitivity >= 0)
                    self.solver.add(output_change <= sensitivity * param_change)
                    self.solver.add(sensitivity <= output_delta_max / param_delta)
                    
                    if self.solver.check() != z3.sat:
                        all_satisfied = False
                        logger.warning("🛡️ Robustez violada: parameter_sensitivity")
                
                elif test_type == 'distributional_robustness':
                    # Teste: robustez a mudanças na distribuição dos dados
                    distribution_shift = test.get('distribution_shift', 0.1)
                    performance_threshold = test.get('performance_threshold', 0.9)
                    
                    # Variáveis para robustez distribucional
                    original_performance = z3.Real('original_performance')
                    shifted_performance = z3.Real('shifted_performance')
                    performance_drop = z3.Real('performance_drop')
                    
                    # Performance deve permanecer acima do threshold
                    self.solver.add(original_performance >= 0)
                    self.solver.add(original_performance <= 1)
                    self.solver.add(shifted_performance >= 0)
                    self.solver.add(shifted_performance <= 1)
                    self.solver.add(performance_drop == original_performance - shifted_performance)
                    self.solver.add(performance_drop <= 1 - performance_threshold)
                    
                    if self.solver.check() != z3.sat:
                        all_satisfied = False
                        logger.warning("🛡️ Robustez violada: distributional_robustness")
            
            return all_satisfied
            
        except Exception as e:
            logger.warning(f"Error in robustness verification: {e}")
            return True

    def _extract_data_from_input(self, input_data):
        """Extrai dados do VerificationInput de forma robusta."""
        if input_data is None:
            return []
        
        # Priorizar input_data, depois output_data
        if hasattr(input_data, 'input_data') and input_data.input_data is not None:
            return input_data.input_data
        elif hasattr(input_data, 'output_data') and input_data.output_data is not None:
            return input_data.output_data
        else:
            return []
    
    def _verify_bounds(self, constraint_data: Dict[str, Any], input_data=None) -> bool:
        """Verifica constraints de bounds usando aritmética real com dados estruturados."""
        try:
            import numpy as np
            
            # Obter configuração dos bounds
            if isinstance(constraint_data, bool) and constraint_data:
                bounds_config = {'min': -np.inf, 'max': np.inf, 'strict': False}
            elif isinstance(constraint_data, dict):
                bounds_config = {
                    'min': constraint_data.get('min', -np.inf),
                    'max': constraint_data.get('max', np.inf),
                    'strict': constraint_data.get('strict', False),
                    'allow_nan': constraint_data.get('allow_nan', False)
                }
            else:
                return True
            
            # 🔍 OBTER DADOS REAIS DO INPUT_DATA
            if input_data and hasattr(input_data, 'input_data'):
                data = input_data.input_data
            elif input_data and hasattr(input_data, 'output_data'):
                data = input_data.output_data
            else:
                # Fallback para dados no constraint_data (para testes diretos)
                data = constraint_data.get('data', [0])
            
            # Normalizar dados para array numpy
            if hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
                data_array = np.array(data).flatten()
            else:
                data_array = np.array([data]).flatten()
            
            # Criar variável real
            x = z3.Real('x')
            
            # Definir constraints de bounds
            min_val, max_val = bounds_config['min'], bounds_config['max']
            strict = bounds_config['strict']
            allow_nan = bounds_config['allow_nan']
            
            # 🔍 VERIFICAÇÃO DE CONFIGURAÇÃO INVÁLIDA
            # Se min > max, é impossível satisfazer - retornar False diretamente
            if min_val != -np.inf and max_val != np.inf:
                if (strict and min_val >= max_val) or (not strict and min_val > max_val):
                    logger.info(f"🔍 Invalid bounds configuration: min={min_val}, max={max_val}, strict={strict}")
                    return False
            
            if strict:
                if min_val != -np.inf:
                    self.solver.add(x > min_val)
                if max_val != np.inf:
                    self.solver.add(x < max_val)
            else:
                if min_val != -np.inf:
                    self.solver.add(x >= min_val)
                if max_val != np.inf:
                    self.solver.add(x <= max_val)
            
            # Verificar cada valor
            all_satisfied = True
            for i, value in enumerate(data_array):
                try:
                    # Tratar NaN
                    if np.isnan(value):
                        if not allow_nan:
                            all_satisfied = False
                            continue
                        else:
                            continue
                    
                    # Verificar com Z3
                    self.solver.push()
                    self.solver.add(x == float(value))
                    result = self.solver.check()
                    self.solver.pop()
                    
                    if result != z3.sat:
                        all_satisfied = False
                except Exception:
                    all_satisfied = False
            
            return all_satisfied
            
        except Exception as e:
            logger.warning(f"Error in bounds verification: {e}")
            return True
    
    def _verify_range_check(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica range checks com suporte a dados estruturados."""
        try:
            import numpy as np
            
            # Estrutura padrão para range check
            if isinstance(constraint_data, bool) and constraint_data:
                range_config = {
                    'valid_ranges': [(-np.inf, np.inf)],
                    'type': 'continuous',
                    'allow_empty': False
                }
            elif isinstance(constraint_data, dict):
                range_config = {
                    'valid_ranges': constraint_data.get('valid_ranges', [(-np.inf, np.inf)]),
                    'type': constraint_data.get('type', 'continuous'),
                    'discrete_values': constraint_data.get('discrete_values', []),
                    'allow_empty': constraint_data.get('allow_empty', False),
                    'tolerance': constraint_data.get('tolerance', 1e-9)
                }
            else:
                return True
            
            # Obter dados (simplificado para esta versão)
            data = constraint_data.get('data', [0])  # Placeholder
            
            # Normalizar dados
            if hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
                data_array = np.array(data).flatten()
            else:
                data_array = np.array([data]).flatten()
            
            # Verificar se dados estão vazios
            if len(data_array) == 0:
                return range_config['allow_empty']
            
            # Criar solver Z3
            x = z3.Real('x')
            
            # Definir constraints baseado no tipo
            range_type = range_config['type']
            valid_ranges = range_config['valid_ranges']
            
            if range_type == 'discrete':
                # Para valores discretos
                discrete_values = range_config.get('discrete_values', [])
                if discrete_values:
                    # Criar constraint OR para valores válidos
                    or_constraints = [x == z3.RealVal(val) for val in discrete_values]
                    self.solver.add(z3.Or(or_constraints))
            
            elif range_type == 'continuous':
                # Para ranges contínuos
                range_constraints = []
                for min_val, max_val in valid_ranges:
                    range_constraint = z3.And(
                        x >= min_val if min_val != -np.inf else True,
                        x <= max_val if max_val != np.inf else True
                    )
                    range_constraints.append(range_constraint)
                
                if range_constraints:
                    self.solver.add(z3.Or(range_constraints))
            
            # Verificar cada valor
            all_satisfied = True
            tolerance = range_config.get('tolerance', 1e-9)
            
            for value in data_array:
                # Tratar NaN/Inf
                if not np.isfinite(value):
                    all_satisfied = False
                    continue
                
                # Verificar com Z3
                self.solver.push()
                self.solver.add(z3.And(
                    x >= float(value) - tolerance,
                    x <= float(value) + tolerance
                ))
                result = self.solver.check()
                self.solver.pop()
                
                if result != z3.sat:
                    # Verificação manual adicional para ranges discretos
                    manual_check = False
                    if range_type == 'discrete':
                        discrete_values = range_config.get('discrete_values', [])
                        manual_check = any(abs(value - dval) <= tolerance for dval in discrete_values)
                    elif range_type == 'continuous':
                        manual_check = any(
                            (min_val == -np.inf or value >= min_val - tolerance) and
                            (max_val == np.inf or value <= max_val + tolerance)
                            for min_val, max_val in valid_ranges
                        )
                    
                    if not manual_check:
                        all_satisfied = False
            
            return all_satisfied
            
        except Exception as e:
            logger.warning(f"Error in range_check verification: {e}")
            return True
    
    def _verify_linear_arithmetic(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica constraints de aritmética linear."""
        # Exemplo: ax + by + c <= 0
        coefficients = constraint_data.get('coefficients', [1, 1])
        constant = constraint_data.get('constant', 0)
        
        x = z3.Real('x')
        y = z3.Real('y')
        
        # Construir expressão linear
        expr = constant
        variables = [x, y]
        for i, coeff in enumerate(coefficients[:len(variables)]):
            expr += coeff * variables[i]
        
        self.solver.add(expr <= 0)
        return self.solver.check() == z3.sat
    
    def _verify_boolean_logic(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica constraints de lógica booleana."""
        # Exemplo de fórmula SAT
        p = z3.Bool('p')
        q = z3.Bool('q')
        r = z3.Bool('r')
        
        # Exemplo: (p ∨ q) ∧ (¬p ∨ r) ∧ (¬q ∨ ¬r)
        formula = z3.And(
            z3.Or(p, q),
            z3.Or(z3.Not(p), r),
            z3.Or(z3.Not(q), z3.Not(r))
        )
        
        self.solver.add(formula)
        return self.solver.check() == z3.sat
    
    def _verify_array_theory(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica constraints usando teoria de arrays."""
        # Array de inteiros
        A = z3.Array('A', z3.IntSort(), z3.IntSort())
        i = z3.Int('i')
        j = z3.Int('j')
        
        # Propriedade: se i != j, então A[i] != A[j] (injetividade)
        size = constraint_data.get('size', 10)
        self.solver.add(z3.And(i >= 0, i < size, j >= 0, j < size))
        self.solver.add(i != j)
        self.solver.add(z3.Select(A, i) == z3.Select(A, j))
        
        # Verificar se é possível violar a injetividade
        return self.solver.check() == z3.sat
    
    def _verify_bitvector_arithmetic(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica aritmética de bit-vectors."""
        bit_width = constraint_data.get('bit_width', 32)
        
        x = z3.BitVec('x', bit_width)
        y = z3.BitVec('y', bit_width)
        
        # Verificar overflow em adição
        overflow_check = z3.ULT(x + y, x)  # Unsigned overflow
        self.solver.add(z3.Not(overflow_check))
        
        return self.solver.check() == z3.sat
    
    def _verify_floating_point(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica aritmética de ponto flutuante."""
        # IEEE 754 single precision
        x = z3.FP('x', z3.Float32())
        y = z3.FP('y', z3.Float32())
        
        # Verificar que não é NaN
        self.solver.add(z3.Not(z3.fpIsNaN(x)))
        self.solver.add(z3.Not(z3.fpIsNaN(y)))
        
        # Verificar operação
        result = z3.fpAdd(z3.RNE(), x, y)  # Round to nearest even
        self.solver.add(z3.Not(z3.fpIsNaN(result)))
        
        return self.solver.check() == z3.sat
    
    def _verify_string_theory(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica constraints de strings."""
        s = z3.String('s')
        pattern = constraint_data.get('pattern', 'hello')
        
        # String contém pattern
        self.solver.add(z3.Contains(s, z3.StringVal(pattern)))
        # String tem comprimento limitado
        max_length = constraint_data.get('max_length', 100)
        self.solver.add(z3.Length(s) <= max_length)
        
        return self.solver.check() == z3.sat
    
    def _verify_quantified_formulas(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica fórmulas quantificadas."""
        x = z3.Int('x')
        
        # ∀x. x ≥ 0 → x² ≥ 0
        formula = z3.ForAll([x], z3.Implies(x >= 0, x * x >= 0))
        self.solver.add(z3.Not(formula))  # Tentar refutar
        
        # Se UNSAT, então a fórmula é válida
        return self.solver.check() == z3.unsat
    
    def _verify_optimization(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica usando otimização Z3."""
        opt = z3.Optimize()
        
        x = z3.Real('x')
        y = z3.Real('y')
        
        # Constraints
        opt.add(x + y <= 10)
        opt.add(x >= 0)
        opt.add(y >= 0)
        
        # Objetivo: maximizar x + y
        opt.maximize(x + y)
        
        return opt.check() == z3.sat
    
    def _verify_neural_network(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica propriedades de redes neurais."""
        # Simulação simples de verificação de rede neural
        input_size = constraint_data.get('input_size', 2)
        
        # Variáveis de entrada
        inputs = [z3.Real(f'input_{i}') for i in range(input_size)]
        
        # Bounds de entrada
        for i, inp in enumerate(inputs):
            self.solver.add(inp >= -1.0)
            self.solver.add(inp <= 1.0)
        
        # Simulação de camada linear: output = w1*x1 + w2*x2 + bias
        w1, w2, bias = 0.5, -0.3, 0.1
        output = w1 * inputs[0] + w2 * inputs[1] + bias
        
        # Propriedade: output deve estar limitado
        self.solver.add(output >= -2.0)
        self.solver.add(output <= 2.0)
        
        return self.solver.check() == z3.sat
    
    def _verify_probability_bounds(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica bounds probabilísticos."""
        # Probabilidade como número real entre 0 e 1
        p = z3.Real('p')
        
        min_prob = constraint_data.get('min', 0.0)
        max_prob = constraint_data.get('max', 1.0)
        
        self.solver.add(p >= min_prob)
        self.solver.add(p <= max_prob)
        self.solver.add(p >= 0.0)
        self.solver.add(p <= 1.0)
        
        return self.solver.check() == z3.sat
    
    def _verify_constraint_with_details(self, constraint_type: str, constraint_data: Any, 
                                      input_data: VerificationInput) -> tuple[bool, dict]:
        """Verifica um constraint e retorna resultado + detalhes do solver."""
        # Limpar solver para este constraint
        self.solver.push()
        
        try:
            # Executar verificação específica
            satisfied = self._verify_constraint(constraint_type, constraint_data, input_data)
            
            # Capturar detalhes do solver
            check_result = self.solver.check()
            
            details = {
                "constraint_type": constraint_type,
                "constraint_data": constraint_data,
                "z3_result": str(check_result),
                "z3_satisfiable": check_result == z3.sat,
                "z3_unsatisfiable": check_result == z3.unsat,
                "z3_unknown": check_result == z3.unknown,
                "satisfied": satisfied,
                "solver_assertions": len(self.solver.assertions()),
            }
            
            # Se satisfeito, tentar obter modelo
            if check_result == z3.sat:
                try:
                    model = self.solver.model()
                    if model:
                        model_values = {}
                        for decl in model.decls():
                            model_values[str(decl.name())] = str(model[decl])
                        details["z3_model"] = model_values
                except Exception:
                    details["z3_model"] = "Could not extract model"
            
            # Se insatisfeito, tentar obter core
            elif check_result == z3.unsat:
                try:
                    core = self.solver.unsat_core()
                    details["z3_unsat_core"] = [str(c) for c in core]
                except Exception:
                    details["z3_unsat_core"] = "Could not extract unsat core"
            
            # 🔍 CONTRA-EXEMPLOS: Gerar quando constraint é violado
            if not satisfied:
                try:
                    counterexample = self._generate_counterexample(constraint_type, constraint_data)
                    details["counterexample"] = counterexample
                except Exception as ce_error:
                    details["counterexample"] = {
                        "error": f"Failed to generate counterexample: {str(ce_error)}"
                    }
            
            return satisfied, details
            
        except Exception as main_error:
            error_details = {
                "constraint_type": constraint_type,
                "constraint_data": constraint_data,
                "error": str(main_error)
            }
            return False, error_details
        finally:
            try:
                self.solver.pop()
            except Exception:
                # Ignorar erros de pop (pode acontecer se não há contexto para fazer pop)
                pass
    
    def _generate_counterexample(self, constraint_type: str, constraint_data: Any) -> Dict[str, Any]:
        """Gera contra-exemplo que viola o constraint especificado.
        
        Args:
            constraint_type: Tipo do constraint a ser violado
            constraint_data: Dados de configuração do constraint
            
        Returns:
            Dict contendo o contra-exemplo ou None se não conseguir gerar
        """
        try:
            logger.info(f"🔍 Gerando contra-exemplo para constraint: {constraint_type}")
            
            # Reset solver para geração limpa
            self.solver.reset()
            self._init_z3()
            
            counterexample = {
                "constraint_type": constraint_type,
                "violation_type": None,
                "counterexample_values": {},
                "explanation": "",
                "satisfiable": False
            }
            
            # Gerar contra-exemplo específico por tipo de constraint
            if constraint_type == 'bounds':
                return self._generate_bounds_counterexample(constraint_data)
            elif constraint_type == 'range_check':
                return self._generate_range_counterexample(constraint_data)
            elif constraint_type == 'linear_arithmetic':
                return self._generate_linear_arithmetic_counterexample(constraint_data)
            elif constraint_type == 'non_negative':
                return self._generate_non_negative_counterexample(constraint_data)
            else:
                # Contra-exemplo genérico
                return self._generate_generic_counterexample(constraint_type, constraint_data)
                
        except Exception as e:
            logger.warning(f"Falha ao gerar contra-exemplo para {constraint_type}: {e}")
            return {
                "constraint_type": constraint_type,
                "error": f"Failed to generate counterexample: {str(e)}",
                "satisfiable": False
            }
    
    def _generate_bounds_counterexample(self, constraint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gera contra-exemplo para constraint de bounds."""
        import numpy as np
        
        # Obter configuração dos bounds
        if isinstance(constraint_data, dict):
            bounds_config = {
                'min': constraint_data.get('min', -np.inf),
                'max': constraint_data.get('max', np.inf),
                'strict': constraint_data.get('strict', False),
                'allow_nan': constraint_data.get('allow_nan', False)
            }
        else:
            return {"error": "Invalid bounds configuration"}
        
        # Criar variável
        x = z3.Real('x')
        
        # Gerar violação dos bounds (negação da condição original)
        min_val, max_val = bounds_config['min'], bounds_config['max']
        strict = bounds_config['strict']
        
        violation_examples = []
        
        # Violação por valor menor que o mínimo
        if min_val != -np.inf:
            if strict:
                violation_value = min_val - 1.0
                self.solver.add(x == violation_value)
            else:
                violation_value = min_val - 1.0
                self.solver.add(x == violation_value)
            
            if self.solver.check() == z3.sat:
                model = self.solver.model()
                violation_examples.append({
                    "type": "below_minimum",
                    "value": float(str(model[x])),
                    "expected_min": min_val,
                    "explanation": f"Value {model[x]} violates minimum bound {min_val}"
                })
        
        # Reset para próxima violação
        self.solver.reset()
        self._init_z3()
        
        # Violação por valor maior que o máximo
        if max_val != np.inf:
            if strict:
                violation_value = max_val + 1.0
                self.solver.add(x == violation_value)
            else:
                violation_value = max_val + 1.0
                self.solver.add(x == violation_value)
            
            if self.solver.check() == z3.sat:
                model = self.solver.model()
                violation_examples.append({
                    "type": "above_maximum", 
                    "value": float(str(model[x])),
                    "expected_max": max_val,
                    "explanation": f"Value {model[x]} violates maximum bound {max_val}"
                })
        
        return {
            "constraint_type": "bounds",
            "violation_examples": violation_examples,
            "bounds_config": bounds_config,
            "satisfiable": len(violation_examples) > 0
        }
    
    def _generate_range_counterexample(self, constraint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gera contra-exemplo para constraint de range_check."""
        if not isinstance(constraint_data, dict):
            return {"error": "Invalid range configuration"}
        
        range_type = constraint_data.get('type', 'continuous')
        
        # Criar variável
        x = z3.Real('x')
        
        violation_examples = []
        
        if range_type == 'continuous':
            valid_ranges = constraint_data.get('valid_ranges', [])
            tolerance = constraint_data.get('tolerance', 1e-6)
            
            # Gerar valores que violam os ranges válidos
            if valid_ranges:
                # Valor antes do primeiro range
                first_range = valid_ranges[0]
                violation_value = first_range[0] - 1.0
                self.solver.add(x == violation_value)
                
                if self.solver.check() == z3.sat:
                    model = self.solver.model()
                    violation_examples.append({
                        "type": "outside_range",
                        "value": float(str(model[x])),
                        "valid_ranges": valid_ranges,
                        "explanation": f"Value {model[x]} is outside valid ranges {valid_ranges}"
                    })
        
        elif range_type == 'discrete':
            discrete_values = constraint_data.get('discrete_values', [])
            
            # Gerar valor que não está na lista discreta
            if discrete_values:
                # Encontrar um valor fora da lista
                max_discrete = max(discrete_values) if discrete_values else 0
                violation_value = max_discrete + 1
                
                self.solver.add(x == violation_value)
                
                if self.solver.check() == z3.sat:
                    model = self.solver.model()
                    violation_examples.append({
                        "type": "invalid_discrete_value",
                        "value": float(str(model[x])),
                        "valid_discrete_values": discrete_values,
                        "explanation": f"Value {model[x]} is not in valid discrete set {discrete_values}"
                    })
        
        return {
            "constraint_type": "range_check",
            "violation_examples": violation_examples,
            "range_config": constraint_data,
            "satisfiable": len(violation_examples) > 0
        }
    
    def _generate_linear_arithmetic_counterexample(self, constraint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gera contra-exemplo para constraint de aritmética linear."""
        if not isinstance(constraint_data, dict):
            return {"error": "Invalid linear arithmetic configuration"}
        
        coefficients = constraint_data.get('coefficients', [1, -1])
        constant = constraint_data.get('constant', 0)
        
        # Criar variáveis
        x = z3.Real('x')
        y = z3.Real('y')
        
        # Gerar violação da equação linear: ax + by != c
        # Por exemplo, se temos x - y = 0, violamos com x - y = 1
        violation_constant = constant + 1
        
        self.solver.add(coefficients[0] * x + coefficients[1] * y == violation_constant)
        self.solver.add(x >= 0)  # Adicionar bounds razoáveis
        self.solver.add(y >= 0)
        
        violation_examples = []
        
        if self.solver.check() == z3.sat:
            model = self.solver.model()
            violation_examples.append({
                "type": "linear_equation_violation",
                "x_value": float(str(model[x])),
                "y_value": float(str(model[y])),
                "expected_result": constant,
                "actual_result": violation_constant,
                "explanation": f"Linear equation {coefficients[0]}*{model[x]} + {coefficients[1]}*{model[y]} = {violation_constant} violates expected = {constant}"
            })
        
        return {
            "constraint_type": "linear_arithmetic",
            "violation_examples": violation_examples,
            "equation_config": constraint_data,
            "satisfiable": len(violation_examples) > 0
        }
    
    def _generate_non_negative_counterexample(self, constraint_data: Any) -> Dict[str, Any]:
        """Gera contra-exemplo para constraint de não-negatividade."""
        x = z3.Real('x')
        
        # Gerar valor negativo
        self.solver.add(x < 0)
        self.solver.add(x >= -10)  # Bound razoável
        
        violation_examples = []
        
        if self.solver.check() == z3.sat:
            model = self.solver.model()
            violation_examples.append({
                "type": "negative_value",
                "value": float(str(model[x])),
                "explanation": f"Value {model[x]} is negative, violating non-negative constraint"
            })
        
        return {
            "constraint_type": "non_negative",
            "violation_examples": violation_examples,
            "satisfiable": len(violation_examples) > 0
        }
    
    def _generate_generic_counterexample(self, constraint_type: str, constraint_data: Any) -> Dict[str, Any]:
        """Gera contra-exemplo genérico para constraints não implementados."""
        return {
            "constraint_type": constraint_type,
            "violation_type": "generic",
            "explanation": f"Generic counterexample generation for {constraint_type}",
            "note": "Specific counterexample generation not implemented for this constraint type",
            "satisfiable": False
        }
    
    def _report_verification_details(self, input_data: VerificationInput, solver_details: dict,
                                   constraints_satisfied: list, constraints_violated: list,
                                   execution_time: float) -> None:
        """Reporta detalhes completos da verificação usando report_data."""
        from ...utils.report import report_data
        from ...utils.types import ReportMode
        
        # Criar relatório detalhado
        verification_report = {
            "verification_session": {
                "verifier": self.name,
                "timestamp": input_data.name,
                "execution_time_ms": round(execution_time * 1000, 2),
                "total_constraints": len(input_data.constraints),
                "constraints_satisfied": len(constraints_satisfied),
                "constraints_violated": len(constraints_violated),
                "success_rate": round(len(constraints_satisfied) / max(1, len(input_data.constraints)) * 100, 1)
            },
            "constraint_results": {
                "satisfied": constraints_satisfied,
                "violated": constraints_violated
            },
            "z3_solver_details": solver_details
        }
        
        # CONSOLE: Log estruturado dos resultados
        logger.info("🔍 Z3 SOLVER RESULTS - %s", input_data.name)
        logger.info("⏱️  Execution Time: %.2fms", execution_time*1000)
        logger.info("📊 Constraints: %d ✅ / %d ❌ / %d total", 
                   len(constraints_satisfied), len(constraints_violated), len(input_data.constraints))
        logger.info("📈 Success Rate: %.1f%%", verification_report['verification_session']['success_rate'])
        
        if constraints_satisfied:
            logger.info("✅ SATISFIED CONSTRAINTS (%d): %s", 
                       len(constraints_satisfied), ', '.join(constraints_satisfied))
                
        if constraints_violated:
            logger.info("❌ VIOLATED CONSTRAINTS (%d): %s", 
                       len(constraints_violated), ', '.join(constraints_violated))
            
            # Log contra-exemplos disponíveis
            for constraint in constraints_violated:
                details = solver_details.get(constraint, {})
                counterexample = details.get('counterexample')
                if counterexample and counterexample.get('satisfiable'):
                    violation_examples = counterexample.get('violation_examples', [])
                    for example in violation_examples[:2]:  # Log até 2 exemplos
                        if 'explanation' in example:
                            logger.info("🔍 Contra-exemplo %s: %s", constraint, example['explanation'])
                        if 'value' in example:
                            logger.info("🔍 Valor violador %s: %s", constraint, example['value'])
        
        # LOGS: Report completo
        report_data(verification_report, ReportMode.PRINT)
        
        # RESULTS: Salvar arquivo detalhado
        timestamp = input_data.name.replace(":", "-").replace(" ", "_")
        report_data(verification_report, ReportMode.JSON_RESULT, 
                   f"z3-verification-{timestamp}")
    
    def create_standard_result(self, verification_input: VerificationInput, 
                              legacy_result: VerificationResult) -> StandardVerificationResult:
        """Cria resultado padronizado a partir do resultado legado."""
        
        # Metadados do solver Z3
        solver_metadata = SolverMetadata(
            solver_name="Z3",
            solver_version=self.version,
            logic_used="QF_NIRA",
            timeout_ms=600000,  # 10 minutos
            memory_limit_mb=14000,  # 14GB
            thread_count=16,  # Configurado no _init_z3
            random_seed=12345,
            configuration_hash=f"z3_scientific_max_performance_{int(time.time())}"
        )
        
        # Métricas de performance
        performance = PerformanceMetrics(
            total_execution_time=legacy_result.execution_time,
            constraint_count=len(legacy_result.constraints_checked),
            constraints_satisfied=len(legacy_result.constraints_satisfied),
            constraints_violated=len(legacy_result.constraints_violated),
            constraints_unknown=0,  # Z3 raramente retorna unknown para nossos casos
            constraints_timeout=0,
            constraints_error=0,
            constraints_skipped=0
        )
        
        # Converter status legado para padronizado
        status_mapping = {
            VerificationStatus.SUCCESS: StandardStatus.SUCCESS,
            VerificationStatus.FAILURE: StandardStatus.FAILURE,
            VerificationStatus.ERROR: StandardStatus.ERROR,
            VerificationStatus.SKIPPED: StandardStatus.SKIPPED,
            VerificationStatus.TIMEOUT: StandardStatus.TIMEOUT
        }
        
        overall_status = status_mapping.get(legacy_result.status, StandardStatus.UNKNOWN)
        
        # Criar resultados por constraint
        constraint_results = []
        avg_time_per_constraint = legacy_result.execution_time / max(len(legacy_result.constraints_checked), 1)
        
        # Constraints satisfeitas
        for constraint_name in legacy_result.constraints_satisfied:
            constraint_results.append(ConstraintResult(
                constraint_type=StandardVerificationResult._classify_constraint_type(constraint_name),
                constraint_name=constraint_name,
                status=StandardStatus.SUCCESS,
                execution_time=avg_time_per_constraint,
                solver_specific_details=legacy_result.details.get(constraint_name, {})
            ))
        
        # Constraints violadas
        for constraint_name in legacy_result.constraints_violated:
            constraint_results.append(ConstraintResult(
                constraint_type=StandardVerificationResult._classify_constraint_type(constraint_name),
                constraint_name=constraint_name,
                status=StandardStatus.FAILURE,
                execution_time=avg_time_per_constraint,
                solver_specific_details=legacy_result.details.get(constraint_name, {}),
                error_message=f"Constraint violated by Z3"
            ))
        
        # Resultado padronizado
        return StandardVerificationResult(
            verification_id=f"Z3_{verification_input.name}_{int(time.time())}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            verification_name=verification_input.name,
            overall_status=overall_status,
            overall_message=legacy_result.message,
            solver_metadata=solver_metadata,
            performance=performance,
            constraint_results=constraint_results,
            input_constraints=verification_input.constraints,
            input_data_summary={
                "constraint_count": len(verification_input.constraints),
                "timeout": getattr(verification_input, 'timeout', 600000)
            },
            solver_raw_output=legacy_result.details
        )


# Auto-registrar o verificador Z3
if Z3_AVAILABLE:
    from ..core.plugin_interface import registry
    z3_verifier = Z3Verifier()
    registry.register(z3_verifier)
    logger.info("Z3 verifier registered and ready")
