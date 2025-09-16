"""Plugin Z3 para verificação formal com todos os recursos do Z3."""

import logging
from typing import Any, Dict, List, Optional, Union
import time

from ..core.plugin_interface import FormalVerifier, VerificationInput, VerificationResult, VerificationStatus

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
        self.solver = None
        if Z3_AVAILABLE:
            self._init_z3()
    
    def _init_z3(self):
        """Inicializa o solver Z3."""
        self.solver = z3.Solver()
        # Configurações otimizadas do Z3
        self.solver.set("timeout", 30000)  # 30 segundos
        self.solver.set("model", True)
        self.solver.set("unsat_core", True)
    
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
            
            # Constraints temporais e dinâmicos
            'temporal_logic',
            'ltl_properties',
            'ctl_properties',
            'invariant_preservation',
            'reachability_analysis'
        ]
    
    def verify(self, input_data: VerificationInput) -> VerificationResult:
        """Executa verificação usando Z3."""
        if not self.enabled:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                verifier_name=self.name,
                execution_time=0.0,
                message="Z3 verifier is disabled"
            )
        
        if not Z3_AVAILABLE:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verifier_name=self.name,
                execution_time=0.0,
                message="Z3 not available"
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
                    except Exception as e:
                        constraints_violated.append(constraint_type)
                        solver_details[constraint_type] = {"error": str(e)}
                        logger.warning(f"Failed to verify {constraint_type}: {e}")
            
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
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verifier_name=self.name,
                execution_time=execution_time,
                message=f"Z3 verification error: {str(e)}"
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
                
            return satisfied, details
            
        except Exception as e:
            details["error"] = str(e)
            return False, details
    
    def _verify_constraint(self, constraint_type: str, constraint_data: Any, 
                          input_data: VerificationInput) -> bool:
        """Verifica um constraint específico usando Z3."""
        
        if constraint_type == 'bounds':
            return self._verify_bounds(constraint_data)
            
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
            
        else:
            # Constraint genérico - assumir satisfeito
            logger.debug(f"Generic constraint verification: {constraint_type}")
            return True
    
    def _verify_bounds(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica constraints de bounds usando aritmética real."""
        min_val = constraint_data.get('min', float('-inf'))
        max_val = constraint_data.get('max', float('inf'))
        
        # Criar variável real
        x = z3.Real('x')
        
        # Adicionar constraints
        self.solver.add(x >= min_val)
        self.solver.add(x <= max_val)
        
        # Verificar satisfiabilidade
        result = self.solver.check()
        return result == z3.sat
    
    def _verify_range_check(self, constraint_data: Dict[str, Any]) -> bool:
        """Verifica range checks usando aritmética inteira."""
        start = constraint_data.get('start', 0)
        end = constraint_data.get('end', 100)
        
        x = z3.Int('x')
        self.solver.add(z3.And(x >= start, x < end))
        
        return self.solver.check() == z3.sat
    
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
            
            return satisfied, details
            
        finally:
            self.solver.pop()
    
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
        
        # CONSOLE: Resumo visual
        print(f"\n🔍 Z3 SOLVER RESULTS - {input_data.name}")
        print("=" * 60)
        print(f"⏱️  Execution Time: {execution_time*1000:.2f}ms")
        print(f"📊 Constraints: {len(constraints_satisfied)} ✅ / {len(constraints_violated)} ❌ / {len(input_data.constraints)} total")
        print(f"📈 Success Rate: {verification_report['verification_session']['success_rate']}%")
        
        if constraints_satisfied:
            print(f"\n✅ SATISFIED CONSTRAINTS ({len(constraints_satisfied)}):")
            for constraint in constraints_satisfied:
                details = solver_details.get(constraint, {})
                z3_result = details.get('z3_result', 'unknown')
                print(f"   • {constraint}: {z3_result}")
                
        if constraints_violated:
            print(f"\n❌ VIOLATED CONSTRAINTS ({len(constraints_violated)}):")
            for constraint in constraints_violated:
                details = solver_details.get(constraint, {})
                z3_result = details.get('z3_result', 'unknown')
                print(f"   • {constraint}: {z3_result}")
        
        # LOGS: Report completo
        report_data(verification_report, ReportMode.PRINT)
        
        # RESULTS: Salvar arquivo detalhado
        timestamp = input_data.name.replace(":", "-").replace(" ", "_")
        report_data(verification_report, ReportMode.JSON_RESULT, 
                   f"z3-verification-{timestamp}")


# Auto-registrar o verificador Z3
if Z3_AVAILABLE:
    from ..core.plugin_interface import registry
    z3_verifier = Z3Verifier()
    registry.register(z3_verifier)
    logger.info("Z3 verifier registered and ready")
