"""Configuração avançada para experimentos multi-solver com dataset Wine."""

import os
from pathlib import Path
from typing import Dict, Any, List

from smart_inference_ai_fusion.utils import (
    VerificationMode, 
    SolverChoice, 
    VerificationConfig,
    set_verification_config
)


class WineAdvancedConfig:
    """Configuração avançada para experimentos com dataset Wine."""
    
    # Configurações de verificação para diferentes cenários
    BASIC_VERIFICATION = VerificationConfig(
        mode=VerificationMode.VERIFICATION,
        solver=SolverChoice.AUTO,
        timeout_per_constraint=15.0,
        timeout_total=120.0,
        parallel_solvers=False,
        detailed_logging=False,
        compare_solvers=False
    )
    
    COMPARISON_MODE = VerificationConfig(
        mode=VerificationMode.VERIFICATION,
        solver=SolverChoice.BOTH,
        timeout_per_constraint=30.0,
        timeout_total=300.0,
        parallel_solvers=True,
        max_workers=2,
        detailed_logging=True,
        compare_solvers=True,
        generate_counterexamples=True
    )
    
    COMPLETE_PIPELINE = VerificationConfig(
        mode=VerificationMode.ALL,
        solver=SolverChoice.BOTH,
        timeout_per_constraint=45.0,
        timeout_total=600.0,
        parallel_solvers=True,
        max_workers=2,
        detailed_logging=True,
        save_results=True,
        compare_solvers=True,
        generate_counterexamples=True
    )
    
    PERFORMANCE_FOCUSED = VerificationConfig(
        mode=VerificationMode.VERIFICATION,
        solver=SolverChoice.Z3,  # Z3 geralmente mais rápido
        timeout_per_constraint=10.0,
        timeout_total=60.0,
        parallel_solvers=False,
        detailed_logging=False,
        save_results=True
    )
    
    RESEARCH_MODE = VerificationConfig(
        mode=VerificationMode.ALL,
        solver=SolverChoice.BOTH,
        timeout_per_constraint=60.0,
        timeout_total=1200.0,  # 20 minutos
        parallel_solvers=True,
        max_workers=2,
        detailed_logging=True,
        save_results=True,
        compare_solvers=True,
        generate_counterexamples=True
    )
    
    @classmethod
    def apply_basic_verification(cls):
        """Aplica configuração básica de verificação."""
        set_verification_config(cls.BASIC_VERIFICATION)
        print("🔹 Configuração aplicada: Verificação Básica")
        print(f"   Solver: {cls.BASIC_VERIFICATION.solver.value}")
        print(f"   Timeout: {cls.BASIC_VERIFICATION.timeout_per_constraint}s")
    
    @classmethod
    def apply_comparison_mode(cls):
        """Aplica configuração para comparação entre solvers."""
        set_verification_config(cls.COMPARISON_MODE)
        print("⚡ Configuração aplicada: Modo Comparação")
        print(f"   Solvers: {cls.COMPARISON_MODE.solver.value}")
        print(f"   Paralelo: {cls.COMPARISON_MODE.parallel_solvers}")
        print(f"   Comparação: {cls.COMPARISON_MODE.compare_solvers}")
    
    @classmethod
    def apply_complete_pipeline(cls):
        """Aplica configuração para pipeline completo."""
        set_verification_config(cls.COMPLETE_PIPELINE)
        print("🎯 Configuração aplicada: Pipeline Completo")
        print(f"   Modo: {cls.COMPLETE_PIPELINE.mode.value}")
        print(f"   Solvers: {cls.COMPLETE_PIPELINE.solver.value}")
        print(f"   Timeout total: {cls.COMPLETE_PIPELINE.timeout_total}s")
    
    @classmethod
    def apply_performance_focused(cls):
        """Aplica configuração focada em performance."""
        set_verification_config(cls.PERFORMANCE_FOCUSED)
        print("🚀 Configuração aplicada: Foco em Performance")
        print(f"   Solver: {cls.PERFORMANCE_FOCUSED.solver.value}")
        print(f"   Timeout: {cls.PERFORMANCE_FOCUSED.timeout_per_constraint}s")
    
    @classmethod
    def apply_research_mode(cls):
        """Aplica configuração para pesquisa científica."""
        set_verification_config(cls.RESEARCH_MODE)
        print("🔬 Configuração aplicada: Modo Pesquisa")
        print(f"   Modo: {cls.RESEARCH_MODE.mode.value}")
        print(f"   Timeout total: {cls.RESEARCH_MODE.timeout_total}s (20 min)")
        print(f"   Comparação: {cls.RESEARCH_MODE.compare_solvers}")


def setup_wine_experiment_environment():
    """Configura ambiente completo para experimentos com Wine."""
    
    # Criar diretórios necessários
    results_dir = Path("results/wine_experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_dir = Path("results/solver_comparison/wine")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path("logs/wine")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print("📁 Diretórios criados:")
    print(f"   Resultados: {results_dir}")
    print(f"   Comparações: {comparison_dir}")
    print(f"   Logs: {logs_dir}")
    
    # Configurar variáveis de ambiente para o experimento
    os.environ['EXPERIMENT_NAME'] = 'wine_multi_solver'
    os.environ['DATASET'] = 'wine'
    os.environ['LOG_LEVEL'] = 'INFO'
    
    print("\n🎛️ Variáveis de ambiente configuradas para Wine dataset")


# Configurações pré-definidas para comandos make específicos
MAKEFILE_CONFIGS = {
    "run-basic": {
        "VERIFICATION_MODE": "basic",
        "description": "Algoritmos básicos sem verificação"
    },
    
    "run-inference": {
        "VERIFICATION_MODE": "inference", 
        "description": "Algoritmos com inferência sintética"
    },
    
    "run-verification": {
        "VERIFICATION_MODE": "verification",
        "VERIFICATION_SOLVER": "auto",
        "description": "Verificação formal com solver automático"
    },
    
    "run-z3": {
        "VERIFICATION_MODE": "verification",
        "VERIFICATION_SOLVER": "z3",
        "VERIFICATION_TIMEOUT_CONSTRAINT": "30",
        "description": "Verificação apenas com Z3"
    },
    
    "run-cvc5": {
        "VERIFICATION_MODE": "verification", 
        "VERIFICATION_SOLVER": "cvc5",
        "VERIFICATION_TIMEOUT_CONSTRAINT": "30",
        "description": "Verificação apenas com CVC5"
    },
    
    "run-both-solvers": {
        "VERIFICATION_MODE": "verification",
        "VERIFICATION_SOLVER": "both",
        "VERIFICATION_PARALLEL": "true",
        "VERIFICATION_COMPARE_SOLVERS": "true",
        "VERIFICATION_TIMEOUT_CONSTRAINT": "45",
        "description": "Comparação paralela entre Z3 e CVC5"
    },
    
    "run-all": {
        "VERIFICATION_MODE": "all",
        "VERIFICATION_SOLVER": "both", 
        "VERIFICATION_PARALLEL": "true",
        "VERIFICATION_COMPARE_SOLVERS": "true",
        "VERIFICATION_TIMEOUT_TOTAL": "600",
        "VERIFICATION_DETAILED_LOG": "true",
        "description": "Pipeline completo com comparação"
    },
    
    "run-advanced": {
        "VERIFICATION_MODE": "all",
        "VERIFICATION_SOLVER": "both",
        "VERIFICATION_PARALLEL": "true", 
        "VERIFICATION_COMPARE_SOLVERS": "true",
        "VERIFICATION_TIMEOUT_TOTAL": "1200",
        "VERIFICATION_DETAILED_LOG": "true",
        "VERIFICATION_GENERATE_COUNTEREXAMPLES": "true",
        "description": "Configuração máxima para pesquisa"
    }
}


def print_available_configurations():
    """Imprime todas as configurações disponíveis."""
    print("="*60)
    print("🎛️ CONFIGURAÇÕES DISPONÍVEIS PARA WINE DATASET")
    print("="*60)
    
    print("\n📊 Configurações de Classe:")
    print("   WineAdvancedConfig.apply_basic_verification()")
    print("   WineAdvancedConfig.apply_comparison_mode()")
    print("   WineAdvancedConfig.apply_complete_pipeline()")
    print("   WineAdvancedConfig.apply_performance_focused()")
    print("   WineAdvancedConfig.apply_research_mode()")
    
    print("\n🔧 Comandos Make Disponíveis:")
    for command, config in MAKEFILE_CONFIGS.items():
        print(f"   make {command} EXP=wine")
        print(f"      → {config['description']}")
        
        # Mostrar principais variáveis
        key_vars = [k for k in config.keys() if k != 'description' and k.startswith('VERIFICATION')]
        if key_vars:
            vars_str = ", ".join([f"{k}={config[k]}" for k in key_vars[:2]])
            if len(key_vars) > 2:
                vars_str += "..."
            print(f"      ({vars_str})")
        print()


if __name__ == "__main__":
    print_available_configurations()