"""Framework para comparação de resultados entre diferentes solvers SMT."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .logging import logger

# Imports do novo sistema de resultados padronizados
try:
    from ..verification.core.result_schema import (
        ResultSchemaManager,
        StandardStatus,
        StandardVerificationResult,
    )

    STANDARD_SCHEMA_AVAILABLE = True
except ImportError:
    STANDARD_SCHEMA_AVAILABLE = False
    logger.warning("Standard result schema not available, using legacy comparison only")


@dataclass
class SolverMetrics:
    """Métricas de performance de um solver."""

    solver_name: str
    total_verifications: int = 0
    successful_verifications: int = 0
    failed_verifications: int = 0
    error_verifications: int = 0
    timeout_verifications: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float("inf")
    max_execution_time: float = 0.0
    constraints_checked: int = 0
    constraints_satisfied: int = 0
    constraints_violated: int = 0

    @property
    def average_execution_time(self) -> float:
        """Tempo médio de execução (calculado sob demanda)."""
        if self.total_verifications == 0:
            return 0.0
        return self.total_execution_time / self.total_verifications

    def update_from_result(self, result: Dict[str, Any]):
        """Atualiza métricas com resultado de uma verificação."""
        self.total_verifications += 1

        status = result.get("status", "ERROR")
        execution_time = result.get("execution_time", 0.0)

        # Contadores por status
        if status == "SUCCESS":
            self.successful_verifications += 1
        elif status == "FAILURE":
            self.failed_verifications += 1
        elif status == "TIMEOUT":
            self.timeout_verifications += 1
        else:
            self.error_verifications += 1

        # Métricas de tempo
        self.total_execution_time += execution_time
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)

        # Métricas de constraints
        self.constraints_checked += len(result.get("constraints_checked", []))
        self.constraints_satisfied += len(result.get("constraints_satisfied", []))
        self.constraints_violated += len(result.get("constraints_violated", []))

    @property
    def success_rate(self) -> float:
        """Taxa de sucesso (0.0 a 1.0)."""
        if self.total_verifications == 0:
            return 0.0
        return self.successful_verifications / self.total_verifications

    @property
    def error_rate(self) -> float:
        """Taxa de erro (0.0 a 1.0)."""
        if self.total_verifications == 0:
            return 0.0
        return self.error_verifications / self.total_verifications

    @property
    def timeout_rate(self) -> float:
        """Taxa de timeout (0.0 a 1.0)."""
        if self.total_verifications == 0:
            return 0.0
        return self.timeout_verifications / self.total_verifications


@dataclass
class ComparisonResult:
    """Resultado de comparação entre solvers."""

    timestamp: str
    experiment_name: str
    solvers_compared: List[str]
    total_comparisons: int
    agreement_rate: float
    performance_winner: str
    reliability_winner: str
    solver_metrics: Dict[str, SolverMetrics]
    detailed_comparisons: List[Dict[str, Any]]
    summary: Dict[str, Any]


class SolverComparison:
    """Framework para comparação sistemática entre solvers SMT."""

    def __init__(self, output_dir: Optional[str] = None):
        """Inicializa o framework de comparação.

        Args:
            output_dir: Diretório para salvar resultados. Se None, usa 'results/solver_comparison'
        """
        self.output_dir = Path(output_dir or "results/solver_comparison")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Métricas acumuladas por solver
        self.solver_metrics: Dict[str, SolverMetrics] = {}

        # Histórico de comparações
        self.comparison_history: List[Dict[str, Any]] = []

        logger.info("📊 Comparison framework initialized: %s", self.output_dir)

    def add_verification_results(self, experiment_name: str, verification_results: Dict[str, Any]):
        """Adiciona resultados de verificação para análise.

        Args:
            experiment_name: Nome do experimento
            verification_results: Resultados da verificação multi-solver
        """
        if not verification_results:
            return

        # Extrair apenas resultados de solvers (ignorar 'comparison')
        solver_results = {
            k: v
            for k, v in verification_results.items()
            if k != "comparison" and isinstance(v, dict)
        }

        if len(solver_results) < 2:
            return  # Não é possível comparar com menos de 2 solvers

        # Atualizar métricas de cada solver
        for solver_name, result in solver_results.items():
            if solver_name not in self.solver_metrics:
                self.solver_metrics[solver_name] = SolverMetrics(solver_name=solver_name)

            self.solver_metrics[solver_name].update_from_result(result)

        # Adicionar ao histórico de comparações
        comparison_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": experiment_name,
            "solvers": list(solver_results.keys()),
            "results": solver_results,
            "agreement": self._analyze_agreement(solver_results),
            "performance": self._analyze_performance(solver_results),
        }

        self.comparison_history.append(comparison_entry)

        logger.info(
            "📊 Added comparison: %s (%d solvers)",
            experiment_name,
            len(solver_results),
        )

    def _analyze_agreement(self, solver_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa concordância entre solvers."""
        statuses = [result.get("status", "ERROR") for result in solver_results.values()]
        unique_statuses = set(statuses)

        return {
            "status_agreement": len(unique_statuses) == 1,
            "common_status": list(unique_statuses)[0] if len(unique_statuses) == 1 else None,
            "status_distribution": {status: statuses.count(status) for status in unique_statuses},
        }

    def _analyze_performance(self, solver_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa performance entre solvers."""
        execution_times = {
            solver: result.get("execution_time", float("inf"))
            for solver, result in solver_results.items()
        }

        if not execution_times:
            return {}

        fastest_solver = min(execution_times, key=execution_times.get)
        slowest_solver = max(execution_times, key=execution_times.get)

        return {
            "fastest_solver": fastest_solver,
            "slowest_solver": slowest_solver,
            "execution_times": execution_times,
            "time_ratio": (
                execution_times[slowest_solver] / execution_times[fastest_solver]
                if execution_times[fastest_solver] > 0
                else float("inf")
            ),
        }

    def generate_comparison_report(
        self, experiment_name: str = "multi_solver_comparison"
    ) -> ComparisonResult:
        """Gera relatório completo de comparação."""
        if not self.comparison_history:
            raise ValueError("Nenhuma comparação disponível para gerar relatório")

        # Análise global
        total_comparisons = len(self.comparison_history)
        agreements = [comp["agreement"]["status_agreement"] for comp in self.comparison_history]
        agreement_rate = sum(agreements) / len(agreements) if agreements else 0.0

        # Análise de performance
        performance_wins = {}
        reliability_wins = {}

        for comparison in self.comparison_history:
            # Winner de performance (mais rápido)
            fastest = comparison["performance"].get("fastest_solver")
            if fastest:
                performance_wins[fastest] = performance_wins.get(fastest, 0) + 1

            # Winner de confiabilidade (status SUCCESS)
            for solver, result in comparison["results"].items():
                if result.get("status") == "SUCCESS":
                    reliability_wins[solver] = reliability_wins.get(solver, 0) + 1

        performance_winner = (
            max(performance_wins, key=performance_wins.get) if performance_wins else "N/A"
        )
        reliability_winner = (
            max(reliability_wins, key=reliability_wins.get) if reliability_wins else "N/A"
        )

        # Criar resultado
        result = ComparisonResult(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            experiment_name=experiment_name,
            solvers_compared=list(self.solver_metrics.keys()),
            total_comparisons=total_comparisons,
            agreement_rate=agreement_rate,
            performance_winner=performance_winner,
            reliability_winner=reliability_winner,
            solver_metrics=dict(self.solver_metrics),
            detailed_comparisons=list(self.comparison_history),
            summary=self._generate_summary(),
        )

        # Salvar relatório
        self._save_comparison_report(result)

        return result

    def _generate_summary(self) -> Dict[str, Any]:
        """Gera resumo estatístico das comparações."""
        if not self.solver_metrics:
            return {}

        summary = {
            "total_solvers": len(self.solver_metrics),
            "total_experiments": len(self.comparison_history),
            "solver_rankings": {},
            "performance_analysis": {},
            "reliability_analysis": {},
        }

        # Ranking por diferentes critérios
        solvers = list(self.solver_metrics.values())

        # Ranking por taxa de sucesso
        success_ranking = sorted(solvers, key=lambda s: s.success_rate, reverse=True)
        summary["solver_rankings"]["success_rate"] = [s.solver_name for s in success_ranking]

        # Ranking por velocidade média
        speed_ranking = sorted(solvers, key=lambda s: s.average_execution_time)
        summary["solver_rankings"]["average_speed"] = [s.solver_name for s in speed_ranking]

        # Ranking por confiabilidade (menor taxa de erro)
        reliability_ranking = sorted(solvers, key=lambda s: s.error_rate)
        summary["solver_rankings"]["reliability"] = [s.solver_name for s in reliability_ranking]

        # Análise detalhada
        for solver_metrics in solvers:
            solver_name = solver_metrics.solver_name
            summary["performance_analysis"][solver_name] = {
                "avg_time": solver_metrics.average_execution_time,
                "min_time": solver_metrics.min_execution_time,
                "max_time": solver_metrics.max_execution_time,
                "total_time": solver_metrics.total_execution_time,
            }

            summary["reliability_analysis"][solver_name] = {
                "success_rate": solver_metrics.success_rate,
                "error_rate": solver_metrics.error_rate,
                "timeout_rate": solver_metrics.timeout_rate,
                "total_verifications": solver_metrics.total_verifications,
            }

        return summary

    def _save_comparison_report(self, result: ComparisonResult):
        """Salva relatório de comparação em diferentes formatos."""
        timestamp_str = result.timestamp.replace(":", "-").replace(" ", "_")
        base_filename = f"comparison_{result.experiment_name}_{timestamp_str}"

        # Salvar JSON completo
        json_file = self.output_dir / f"{base_filename}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Salvar CSV com métricas resumidas
        self._save_metrics_csv(result, base_filename)

        # Salvar relatório de texto
        self._save_text_report(result, base_filename)

        logger.info("📊 Report saved: %s (JSON, CSV, TXT)", base_filename)

    def _save_metrics_csv(self, result: ComparisonResult, base_filename: str):
        """Salva métricas em formato CSV."""
        metrics_data = []

        for solver_name, metrics in result.solver_metrics.items():
            metrics_data.append(
                {
                    "solver": solver_name,
                    "total_verifications": metrics.total_verifications,
                    "success_rate": metrics.success_rate,
                    "error_rate": metrics.error_rate,
                    "timeout_rate": metrics.timeout_rate,
                    "avg_execution_time": metrics.average_execution_time,
                    "min_execution_time": metrics.min_execution_time,
                    "max_execution_time": metrics.max_execution_time,
                    "constraints_checked": metrics.constraints_checked,
                    "constraints_satisfied": metrics.constraints_satisfied,
                    "constraints_violated": metrics.constraints_violated,
                }
            )

        df = pd.DataFrame(metrics_data)
        csv_file = self.output_dir / f"{base_filename}_metrics.csv"
        df.to_csv(csv_file, index=False)

    def _save_text_report(self, result: ComparisonResult, base_filename: str):
        """Salva relatório em formato texto legível."""
        txt_file = self.output_dir / f"{base_filename}_report.txt"

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO DE COMPARAÇÃO MULTI-SOLVER\n")
            f.write(f"Experimento: {result.experiment_name}\n")
            f.write(f"Data/Hora: {result.timestamp}\n")
            f.write("=" * 80 + "\n\n")

            f.write("RESUMO GERAL:\n")
            f.write(f"- Solvers comparados: {', '.join(result.solvers_compared)}\n")
            f.write(f"- Total de comparações: {result.total_comparisons}\n")
            f.write(f"- Taxa de concordância: {result.agreement_rate:.2%}\n")
            f.write(f"- Vencedor em performance: {result.performance_winner}\n")
            f.write(f"- Vencedor em confiabilidade: {result.reliability_winner}\n\n")

            f.write("MÉTRICAS POR SOLVER:\n")
            f.write("-" * 50 + "\n")
            for solver_name, metrics in result.solver_metrics.items():
                f.write(f"\n{solver_name}:\n")
                f.write(f"  Verificações totais: {metrics.total_verifications}\n")
                f.write(f"  Taxa de sucesso: {metrics.success_rate:.2%}\n")
                f.write(f"  Taxa de erro: {metrics.error_rate:.2%}\n")
                f.write(f"  Taxa de timeout: {metrics.timeout_rate:.2%}\n")
                f.write(f"  Tempo médio: {metrics.average_execution_time:.4f}s\n")
                f.write(
                    f"  Tempo mín/máx: "
                    f"{metrics.min_execution_time:.4f}s / "
                    f"{metrics.max_execution_time:.4f}s\n"
                )

            # Rankings
            f.write("\n\nRANKINGS:\n")
            f.write("-" * 30 + "\n")

            if "success_rate" in result.summary.get("solver_rankings", {}):
                f.write(
                    f"Taxa de Sucesso: "
                    f"{' > '.join(result.summary['solver_rankings']['success_rate'])}\n"
                )

            if "average_speed" in result.summary.get("solver_rankings", {}):
                f.write(
                    f"Velocidade Média: "
                    f"{' > '.join(result.summary['solver_rankings']['average_speed'])}\n"
                )

            if "reliability" in result.summary.get("solver_rankings", {}):
                f.write(
                    f"Confiabilidade: "
                    f"{' > '.join(result.summary['solver_rankings']['reliability'])}\n"
                )

    def print_summary(self):
        """Imprime resumo das comparações atuais usando logging."""
        if not self.solver_metrics:
            logger.info("📊 No comparisons available yet.")
            return

        logger.info("=" * 60)
        logger.info("📊 MULTI-SOLVER COMPARISON SUMMARY")
        logger.info("=" * 60)

        logger.info("Total solvers: %d", len(self.solver_metrics))
        logger.info("Total experiments: %d", len(self.comparison_history))

        logger.info("METRICS PER SOLVER:")
        logger.info("-" * 40)
        for solver_name, metrics in self.solver_metrics.items():
            logger.info("\n%s:", solver_name)
            logger.info(
                "  ✅ Successes: %d/%d (%.1f%%)",
                metrics.successful_verifications,
                metrics.total_verifications,
                metrics.success_rate * 100,
            )
            logger.info("  ⏱️ Average time: %.4fs", metrics.average_execution_time)
            logger.info("  ❌ Error rate: %.1f%%", metrics.error_rate * 100)


class StandardizedSolverComparison:
    """Comparação avançada usando resultados padronizados."""

    def __init__(self):
        self.standard_results: List[StandardVerificationResult] = []
        self.schema_manager = ResultSchemaManager() if STANDARD_SCHEMA_AVAILABLE else None

    def add_standard_result(self, result: StandardVerificationResult):
        """Adiciona resultado padronizado."""
        if not STANDARD_SCHEMA_AVAILABLE:
            logger.error("Standard schema not available")
            return

        self.standard_results.append(result)
        logger.info("📊 Added standard result from %s", result.solver_metadata.solver_name)

    def add_legacy_result(self, legacy_result, solver_name: str, solver_version: str = "unknown"):
        """Converte e adiciona resultado legado."""
        if not STANDARD_SCHEMA_AVAILABLE:
            logger.error("Standard schema not available")
            return

        standard_result = self.schema_manager.standardize_result(
            legacy_result, solver_name, solver_version
        )
        self.add_standard_result(standard_result)

    def generate_advanced_comparison(self) -> Dict[str, Any]:
        """Gera comparação avançada usando resultados padronizados."""
        if not STANDARD_SCHEMA_AVAILABLE or not self.standard_results:
            return {"error": "No standard results available"}

        comparison = self.schema_manager.compare_results(self.standard_results)

        # Adicionar análises específicas
        comparison["detailed_analysis"] = self._generate_detailed_analysis()
        comparison["constraint_type_analysis"] = self._analyze_by_constraint_type()
        comparison["performance_benchmarks"] = self._generate_performance_benchmarks()
        comparison["reliability_scores"] = self._calculate_reliability_scores()

        return comparison

    def _generate_detailed_analysis(self) -> Dict[str, Any]:
        """Gera análise detalhada dos resultados."""
        analysis = {
            "solver_count": len(set(r.solver_metadata.solver_name for r in self.standard_results)),
            "total_experiments": len(self.standard_results),
            "constraint_coverage": {},
            "performance_distribution": {},
            "error_patterns": {},
        }

        # Análise de cobertura de constraints
        all_constraints = set()
        for result in self.standard_results:
            for constraint_result in result.constraint_results:
                all_constraints.add(constraint_result.constraint_type)

        analysis["constraint_coverage"]["total_types"] = len(all_constraints)
        analysis["constraint_coverage"]["types"] = list(all_constraints)

        # Distribuição de performance por solver
        for result in self.standard_results:
            solver_name = result.solver_metadata.solver_name
            if solver_name not in analysis["performance_distribution"]:
                analysis["performance_distribution"][solver_name] = []
            analysis["performance_distribution"][solver_name].append(
                result.performance.total_execution_time
            )

        return analysis

    def _analyze_by_constraint_type(self) -> Dict[str, Any]:
        """Analisa performance por tipo de constraint."""
        type_analysis = {}

        for result in self.standard_results:
            solver_name = result.solver_metadata.solver_name

            for constraint_result in result.constraint_results:
                constraint_type = constraint_result.constraint_type

                if constraint_type not in type_analysis:
                    type_analysis[constraint_type] = {}

                if solver_name not in type_analysis[constraint_type]:
                    type_analysis[constraint_type][solver_name] = {
                        "total": 0,
                        "success": 0,
                        "failure": 0,
                        "avg_time": 0.0,
                        "times": [],
                    }

                stats = type_analysis[constraint_type][solver_name]
                stats["total"] += 1
                stats["times"].append(constraint_result.execution_time)

                if constraint_result.status == StandardStatus.SUCCESS:
                    stats["success"] += 1
                elif constraint_result.status == StandardStatus.FAILURE:
                    stats["failure"] += 1

        # Calcular médias
        for constraint_type, solvers in type_analysis.items():
            for solver_name, stats in solvers.items():
                if stats["times"]:
                    stats["avg_time"] = sum(stats["times"]) / len(stats["times"])
                    stats["success_rate"] = stats["success"] / stats["total"]
                    del stats["times"]  # Remover dados raw para economizar espaço

        return type_analysis

    def _generate_performance_benchmarks(self) -> Dict[str, Any]:
        """Gera benchmarks de performance."""
        benchmarks = {}

        for result in self.standard_results:
            solver_name = result.solver_metadata.solver_name

            if solver_name not in benchmarks:
                benchmarks[solver_name] = {
                    "execution_times": [],
                    "success_rates": [],
                    "constraint_counts": [],
                    "memory_usage": [],
                    "thread_counts": [],
                }

            bench = benchmarks[solver_name]
            bench["execution_times"].append(result.performance.total_execution_time)
            bench["success_rates"].append(result.performance.success_rate)
            bench["constraint_counts"].append(result.performance.constraint_count)
            bench["thread_counts"].append(result.solver_metadata.thread_count)

            if result.performance.memory_usage_mb:
                bench["memory_usage"].append(result.performance.memory_usage_mb)

        # Calcular estatísticas
        for solver_name, data in benchmarks.items():
            for metric, values in data.items():
                if values and isinstance(values[0], (int, float)):
                    data[f"{metric}_stats"] = {
                        "mean": np.mean(values) if values else 0,
                        "std": np.std(values) if len(values) > 1 else 0,
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                        "count": len(values),
                    }

        return benchmarks

    def _calculate_reliability_scores(self) -> Dict[str, float]:
        """Calcula scores de confiabilidade por solver."""
        reliability_scores = {}

        for result in self.standard_results:
            solver_name = result.solver_metadata.solver_name

            if solver_name not in reliability_scores:
                reliability_scores[solver_name] = {
                    "completion_rates": [],
                    "success_rates": [],
                    "error_rates": [],
                    "timeout_rates": [],
                }

            scores = reliability_scores[solver_name]
            scores["completion_rates"].append(result.performance.completion_rate)
            scores["success_rates"].append(result.performance.success_rate)

            # Taxa de erro
            total = result.performance.constraint_count
            if total > 0:
                error_rate = result.performance.constraints_error / total
                timeout_rate = result.performance.constraints_timeout / total
                scores["error_rates"].append(error_rate)
                scores["timeout_rates"].append(timeout_rate)

        # Calcular scores finais
        final_scores = {}
        for solver_name, data in reliability_scores.items():
            final_scores[solver_name] = {
                "overall_reliability": np.mean(data["completion_rates"]) * 0.4
                + np.mean(data["success_rates"]) * 0.4
                + (1.0 - np.mean(data["error_rates"])) * 0.2,
                "avg_completion_rate": np.mean(data["completion_rates"]),
                "avg_success_rate": np.mean(data["success_rates"]),
                "avg_error_rate": np.mean(data["error_rates"]),
                "avg_timeout_rate": np.mean(data["timeout_rates"]),
            }

        return final_scores

    def export_standard_results(
        self, output_dir: str = "results/standard_comparison"
    ) -> Dict[str, str]:
        """Exporta resultados padronizados em múltiplos formatos."""
        if not STANDARD_SCHEMA_AVAILABLE:
            return {"error": "Standard schema not available"}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        exported_files = {}

        try:
            # JSON detalhado
            json_file = output_path / f"standard_results_{timestamp}.json"
            json_content = self.schema_manager.export_results(self.standard_results, "json")
            json_file.write_text(json_content, encoding="utf-8")
            exported_files["json"] = str(json_file)

            # CSV resumido
            csv_file = output_path / f"standard_summary_{timestamp}.csv"
            csv_content = self.schema_manager.export_results(self.standard_results, "csv")
            csv_file.write_text(csv_content, encoding="utf-8")
            exported_files["csv"] = str(csv_file)

            # Relatório Markdown
            md_file = output_path / f"standard_report_{timestamp}.md"
            md_content = self.schema_manager.export_results(self.standard_results, "markdown")
            md_file.write_text(md_content, encoding="utf-8")
            exported_files["markdown"] = str(md_file)

            # Comparação avançada
            comparison_file = output_path / f"advanced_comparison_{timestamp}.json"
            comparison_data = self.generate_advanced_comparison()
            comparison_file.write_text(
                json.dumps(comparison_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            exported_files["comparison"] = str(comparison_file)

            logger.info("Standard results exported to %s", output_dir)
            return exported_files

        except OSError as e:
            logger.error("Failed to export standard results: %s", e)
            return {"error": str(e)}


# Instâncias globais
global_comparison = SolverComparison()
global_standard_comparison = StandardizedSolverComparison() if STANDARD_SCHEMA_AVAILABLE else None


def add_comparison_result(experiment_name: str, verification_results: Dict[str, Any]):
    """Adiciona resultado de comparação à instância global."""
    global_comparison.add_verification_results(experiment_name, verification_results)


def add_standard_result(result):
    """Adiciona resultado padronizado à comparação global."""
    if global_standard_comparison and STANDARD_SCHEMA_AVAILABLE:
        if hasattr(result, "solver_metadata"):  # É StandardVerificationResult
            global_standard_comparison.add_standard_result(result)
        else:  # É resultado legado, precisa converter
            # Tentar extrair nome do solver
            solver_name = getattr(result, "verifier_name", "unknown")
            global_standard_comparison.add_legacy_result(result, solver_name)


def generate_comparison_report(
    experiment_name: str = "multi_solver_comparison",
) -> ComparisonResult:
    """Gera relatório usando a instância global."""
    return global_comparison.generate_comparison_report(experiment_name)


def generate_advanced_comparison_report() -> Dict[str, Any]:
    """Gera relatório avançado usando resultados padronizados."""
    if not global_standard_comparison:
        return {"error": "Standard comparison not available"}
    return global_standard_comparison.generate_advanced_comparison()


def print_comparison_summary():
    """Imprime resumo usando a instância global."""
    global_comparison.print_summary()
