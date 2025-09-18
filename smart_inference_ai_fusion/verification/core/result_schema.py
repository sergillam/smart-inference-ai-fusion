"""Esquema padronizado de resultados para verificação formal multi-solver."""

import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json


class StandardStatus(Enum):
    """Status padronizado entre todos os solvers."""
    SUCCESS = "success"          # Todas as constraints satisfeitas
    FAILURE = "failure"          # Pelo menos uma constraint violada
    UNKNOWN = "unknown"          # Solver não conseguiu determinar
    TIMEOUT = "timeout"          # Timeout atingido
    ERROR = "error"             # Erro durante verificação
    SKIPPED = "skipped"         # Verificação pulada
    UNSUPPORTED = "unsupported" # Constraint não suportado pelo solver


class ConstraintType(Enum):
    """Tipos de constraints padronizados."""
    BOUNDS = "bounds"
    LINEAR = "linear_arithmetic"
    NONLINEAR = "nonlinear_arithmetic"
    LOGICAL = "logical_constraint"
    TEMPORAL = "temporal_constraint"
    PROBABILISTIC = "probabilistic_constraint"
    ML_SPECIFIC = "ml_specific"
    OPTIMIZATION = "optimization"
    CUSTOM = "custom"


@dataclass
class ConstraintResult:
    """Resultado padronizado para uma constraint individual."""
    constraint_type: str
    constraint_name: str
    status: StandardStatus
    execution_time: float
    solver_specific_details: Dict[str, Any] = None
    error_message: Optional[str] = None
    satisfying_assignment: Optional[Dict[str, Any]] = None
    unsatisfiable_core: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.solver_specific_details is None:
            self.solver_specific_details = {}


@dataclass
class SolverMetadata:
    """Metadados do solver para reprodutibilidade."""
    solver_name: str
    solver_version: str
    logic_used: str
    timeout_ms: int
    memory_limit_mb: int
    thread_count: int
    random_seed: Optional[int] = None
    configuration_hash: Optional[str] = None
    system_info: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.system_info is None:
            import platform
            import os
            self.system_info = {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "python_version": platform.python_version()
            }


@dataclass
class PerformanceMetrics:
    """Métricas de performance padronizadas."""
    total_execution_time: float
    constraint_count: int
    constraints_satisfied: int
    constraints_violated: int
    constraints_unknown: int
    constraints_timeout: int
    constraints_error: int
    constraints_skipped: int
    memory_usage_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    restart_count: Optional[int] = None
    decisions_count: Optional[int] = None
    propagations_count: Optional[int] = None
    conflicts_count: Optional[int] = None
    
    @property
    def success_rate(self) -> float:
        """Taxa de sucesso das constraints."""
        if self.constraint_count == 0:
            return 0.0
        return self.constraints_satisfied / self.constraint_count
    
    @property
    def completion_rate(self) -> float:
        """Taxa de conclusão (não timeout/error/skipped)."""
        if self.constraint_count == 0:
            return 0.0
        completed = self.constraints_satisfied + self.constraints_violated + self.constraints_unknown
        return completed / self.constraint_count


@dataclass
class StandardVerificationResult:
    """Resultado de verificação padronizado entre todos os solvers."""
    
    # === IDENTIFICAÇÃO ===
    verification_id: str
    timestamp: str
    verification_name: str
    
    # === STATUS GERAL ===
    overall_status: StandardStatus
    overall_message: str
    
    # === METADADOS DO SOLVER ===
    solver_metadata: SolverMetadata
    
    # === PERFORMANCE ===
    performance: PerformanceMetrics
    
    # === RESULTADOS POR CONSTRAINT ===
    constraint_results: List[ConstraintResult]
    
    # === DADOS DE ENTRADA ===
    input_constraints: Dict[str, Any] = None
    input_data_summary: Dict[str, Any] = None
    
    # === ANÁLISE COMPARATIVA (quando aplicável) ===
    comparison_data: Optional[Dict[str, Any]] = None
    
    # === DADOS RAW DO SOLVER (para debugging) ===
    solver_raw_output: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.input_constraints is None:
            self.input_constraints = {}
        if self.input_data_summary is None:
            self.input_data_summary = {}
    
    @classmethod
    def from_legacy_result(cls, legacy_result, solver_name: str, solver_version: str = "unknown") -> 'StandardVerificationResult':
        """Converte resultado legado para formato padronizado."""
        
        # Mapear status legado para padronizado
        status_mapping = {
            "SUCCESS": StandardStatus.SUCCESS,
            "FAILURE": StandardStatus.FAILURE,
            "ERROR": StandardStatus.ERROR,
            "SKIPPED": StandardStatus.SKIPPED,
            "TIMEOUT": StandardStatus.TIMEOUT
        }
        
        overall_status = status_mapping.get(
            getattr(legacy_result, 'status', 'UNKNOWN').upper(), 
            StandardStatus.UNKNOWN
        )
        
        # Extrair informações de performance
        execution_time = getattr(legacy_result, 'execution_time', 0.0)
        constraints_checked = getattr(legacy_result, 'constraints_checked', [])
        constraints_satisfied = getattr(legacy_result, 'constraints_satisfied', [])
        constraints_violated = getattr(legacy_result, 'constraints_violated', [])
        
        # Criar metadados do solver
        solver_metadata = SolverMetadata(
            solver_name=solver_name,
            solver_version=solver_version,
            logic_used="QF_NIRA",  # Padrão do projeto
            timeout_ms=600000,     # 10 minutos padrão
            memory_limit_mb=12000, # 12GB padrão
            thread_count=16        # Padrão
        )
        
        # Criar métricas de performance
        performance = PerformanceMetrics(
            total_execution_time=execution_time,
            constraint_count=len(constraints_checked),
            constraints_satisfied=len(constraints_satisfied),
            constraints_violated=len(constraints_violated),
            constraints_unknown=0,
            constraints_timeout=0,
            constraints_error=0,
            constraints_skipped=0
        )
        
        # Converter constraints para formato padronizado
        constraint_results = []
        
        # Constraints satisfeitas
        for constraint_name in constraints_satisfied:
            constraint_results.append(ConstraintResult(
                constraint_type=cls._classify_constraint_type(constraint_name),
                constraint_name=constraint_name,
                status=StandardStatus.SUCCESS,
                execution_time=execution_time / len(constraints_checked) if constraints_checked else 0.0
            ))
        
        # Constraints violadas
        for constraint_name in constraints_violated:
            constraint_results.append(ConstraintResult(
                constraint_type=cls._classify_constraint_type(constraint_name),
                constraint_name=constraint_name,
                status=StandardStatus.FAILURE,
                execution_time=execution_time / len(constraints_checked) if constraints_checked else 0.0
            ))
        
        return cls(
            verification_id=f"{solver_name}_{int(time.time())}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            verification_name=getattr(legacy_result, 'verifier_name', 'unknown'),
            overall_status=overall_status,
            overall_message=getattr(legacy_result, 'message', ''),
            solver_metadata=solver_metadata,
            performance=performance,
            constraint_results=constraint_results,
            solver_raw_output=getattr(legacy_result, 'details', {})
        )
    
    @staticmethod
    def _classify_constraint_type(constraint_name: str) -> str:
        """Classifica tipo de constraint baseado no nome."""
        constraint_name_lower = constraint_name.lower()
        
        if 'bounds' in constraint_name_lower or 'range' in constraint_name_lower:
            return ConstraintType.BOUNDS.value
        elif 'linear' in constraint_name_lower:
            return ConstraintType.LINEAR.value
        elif 'nonlinear' in constraint_name_lower or 'polynomial' in constraint_name_lower:
            return ConstraintType.NONLINEAR.value
        elif 'probability' in constraint_name_lower or 'distribution' in constraint_name_lower:
            return ConstraintType.PROBABILISTIC.value
        elif any(ml_term in constraint_name_lower for ml_term in ['neural', 'tree', 'model', 'ml', 'ai']):
            return ConstraintType.ML_SPECIFIC.value
        elif 'optimization' in constraint_name_lower or 'minimize' in constraint_name_lower or 'maximize' in constraint_name_lower:
            return ConstraintType.OPTIMIZATION.value
        else:
            return ConstraintType.CUSTOM.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável."""
        result = asdict(self)
        
        # Converter enums para strings
        result['overall_status'] = self.overall_status.value
        
        for constraint_result in result['constraint_results']:
            constraint_result['status'] = constraint_result['status'].value
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Converte para JSON formatado."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo executivo dos resultados."""
        return {
            "solver": self.solver_metadata.solver_name,
            "version": self.solver_metadata.solver_version,
            "status": self.overall_status.value,
            "execution_time": self.performance.total_execution_time,
            "constraints_total": self.performance.constraint_count,
            "constraints_satisfied": self.performance.constraints_satisfied,
            "constraints_violated": self.performance.constraints_violated,
            "success_rate": self.performance.success_rate,
            "completion_rate": self.performance.completion_rate,
            "timestamp": self.timestamp
        }
    
    def get_constraint_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Retorna breakdown de constraints por tipo e status."""
        breakdown = {}
        
        for constraint_result in self.constraint_results:
            constraint_type = constraint_result.constraint_type
            status = constraint_result.status.value
            
            if constraint_type not in breakdown:
                breakdown[constraint_type] = {}
            
            if status not in breakdown[constraint_type]:
                breakdown[constraint_type][status] = 0
            
            breakdown[constraint_type][status] += 1
        
        return breakdown
    
    def compare_with(self, other: 'StandardVerificationResult') -> Dict[str, Any]:
        """Compara com outro resultado padronizado."""
        comparison = {
            "solvers": [self.solver_metadata.solver_name, other.solver_metadata.solver_name],
            "execution_time_comparison": {
                self.solver_metadata.solver_name: self.performance.total_execution_time,
                other.solver_metadata.solver_name: other.performance.total_execution_time,
                "faster_solver": self.solver_metadata.solver_name if self.performance.total_execution_time < other.performance.total_execution_time else other.solver_metadata.solver_name,
                "speedup": abs(self.performance.total_execution_time - other.performance.total_execution_time) / max(self.performance.total_execution_time, other.performance.total_execution_time)
            },
            "accuracy_comparison": {
                self.solver_metadata.solver_name: self.performance.success_rate,
                other.solver_metadata.solver_name: other.performance.success_rate,
                "more_accurate": self.solver_metadata.solver_name if self.performance.success_rate > other.performance.success_rate else other.solver_metadata.solver_name
            },
            "constraint_agreement": self._calculate_constraint_agreement(other),
            "overall_winner": self._determine_overall_winner(other)
        }
        
        return comparison
    
    def _calculate_constraint_agreement(self, other: 'StandardVerificationResult') -> Dict[str, Any]:
        """Calcula concordância entre constraints."""
        my_results = {cr.constraint_name: cr.status for cr in self.constraint_results}
        other_results = {cr.constraint_name: cr.status for cr in other.constraint_results}
        
        common_constraints = set(my_results.keys()) & set(other_results.keys())
        
        if not common_constraints:
            return {"agreement_rate": 0.0, "common_constraints": 0}
        
        agreements = sum(1 for constraint in common_constraints 
                        if my_results[constraint] == other_results[constraint])
        
        return {
            "agreement_rate": agreements / len(common_constraints),
            "common_constraints": len(common_constraints),
            "agreements": agreements,
            "disagreements": len(common_constraints) - agreements
        }
    
    def _determine_overall_winner(self, other: 'StandardVerificationResult') -> str:
        """Determina vencedor geral baseado em critérios ponderados."""
        # Critérios: 40% accuracy, 30% speed, 20% completion, 10% stability
        
        my_score = (
            self.performance.success_rate * 0.4 +
            (1.0 / (self.performance.total_execution_time + 1)) * 0.3 +
            self.performance.completion_rate * 0.2 +
            (1.0 if self.overall_status == StandardStatus.SUCCESS else 0.0) * 0.1
        )
        
        other_score = (
            other.performance.success_rate * 0.4 +
            (1.0 / (other.performance.total_execution_time + 1)) * 0.3 +
            other.performance.completion_rate * 0.2 +
            (1.0 if other.overall_status == StandardStatus.SUCCESS else 0.0) * 0.1
        )
        
        if my_score > other_score:
            return self.solver_metadata.solver_name
        elif other_score > my_score:
            return other.solver_metadata.solver_name
        else:
            return "tie"


class ResultSchemaManager:
    """Gerenciador para conversão e padronização de resultados."""
    
    @staticmethod
    def standardize_result(legacy_result, solver_name: str, solver_version: str = "unknown") -> StandardVerificationResult:
        """Padroniza resultado legado."""
        return StandardVerificationResult.from_legacy_result(legacy_result, solver_name, solver_version)
    
    @staticmethod
    def compare_results(results: List[StandardVerificationResult]) -> Dict[str, Any]:
        """Compara múltiplos resultados padronizados."""
        if len(results) < 2:
            return {"error": "At least 2 results required for comparison"}
        
        comparison = {
            "solvers": [r.solver_metadata.solver_name for r in results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "comparison_matrix": {},
            "ranking": {},
            "summary": {}
        }
        
        # Comparação par a par
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results):
                if i < j:  # Evitar duplicatas
                    key = f"{result1.solver_metadata.solver_name}_vs_{result2.solver_metadata.solver_name}"
                    comparison["comparison_matrix"][key] = result1.compare_with(result2)
        
        # Ranking geral
        solver_scores = {}
        for result in results:
            score = (
                result.performance.success_rate * 0.4 +
                (1.0 / (result.performance.total_execution_time + 1)) * 0.3 +
                result.performance.completion_rate * 0.2 +
                (1.0 if result.overall_status == StandardStatus.SUCCESS else 0.0) * 0.1
            )
            solver_scores[result.solver_metadata.solver_name] = score
        
        # Ordenar por score
        ranked_solvers = sorted(solver_scores.items(), key=lambda x: x[1], reverse=True)
        comparison["ranking"] = {
            "by_overall_score": ranked_solvers,
            "by_accuracy": sorted([(r.solver_metadata.solver_name, r.performance.success_rate) for r in results], key=lambda x: x[1], reverse=True),
            "by_speed": sorted([(r.solver_metadata.solver_name, r.performance.total_execution_time) for r in results], key=lambda x: x[1]),
            "by_completion": sorted([(r.solver_metadata.solver_name, r.performance.completion_rate) for r in results], key=lambda x: x[1], reverse=True)
        }
        
        # Resumo estatístico
        comparison["summary"] = {
            "best_overall": ranked_solvers[0][0] if ranked_solvers else "unknown",
            "fastest": min(results, key=lambda r: r.performance.total_execution_time).solver_metadata.solver_name,
            "most_accurate": max(results, key=lambda r: r.performance.success_rate).solver_metadata.solver_name,
            "most_complete": max(results, key=lambda r: r.performance.completion_rate).solver_metadata.solver_name,
            "average_execution_time": sum(r.performance.total_execution_time for r in results) / len(results),
            "average_success_rate": sum(r.performance.success_rate for r in results) / len(results)
        }
        
        return comparison
    
    @staticmethod
    def export_results(results: List[StandardVerificationResult], format: str = "json", 
                      output_path: str = None) -> str:
        """Exporta resultados em formato específico."""
        
        if format.lower() == "json":
            data = [result.to_dict() for result in results]
            content = json.dumps(data, indent=2, ensure_ascii=False)
        
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if results:
                fieldnames = list(results[0].get_summary().keys())
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow(result.get_summary())
            content = output.getvalue()
            
        elif format.lower() == "markdown":
            content = "# Verification Results Report\n\n"
            for result in results:
                summary = result.get_summary()
                content += f"## {summary['solver']} v{summary['version']}\n"
                content += f"- **Status**: {summary['status']}\n"
                content += f"- **Execution Time**: {summary['execution_time']:.3f}s\n"
                content += f"- **Success Rate**: {summary['success_rate']:.1%}\n"
                content += f"- **Constraints**: {summary['constraints_satisfied']}/{summary['constraints_total']} satisfied\n\n"
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return output_path
        
        return content