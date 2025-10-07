"""Configurações para verificação formal multi-solver."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from .logging import logger


class VerificationMode(Enum):
    """Modos de execução de experimentos."""

    BASIC = "basic"  # Algoritmos normais (sem verificação)
    INFERENCE = "inference"  # Algoritmos com inferência sintética
    VERIFICATION = "verification"  # Algoritmos com verificação formal
    ALL = "all"  # Execução completa (basic + inference + verification)


class SolverChoice(Enum):
    """Escolha de solvers para verificação."""

    Z3 = "z3"  # Apenas Z3
    CVC5 = "cvc5"  # Apenas CVC5
    BOTH = "both"  # Ambos os solvers
    AUTO = "auto"  # Escolha automática (prioridade Z3)


@dataclass
class VerificationConfig:
    """Configuração para verificação formal."""

    # Modo de execução
    mode: VerificationMode = VerificationMode.BASIC

    # Escolha de solver
    solver: SolverChoice = SolverChoice.AUTO

    # Configurações de timeout
    timeout_per_constraint: float = 30.0  # segundos
    timeout_total: float = 300.0  # 5 minutos total

    # Configurações de paralelização
    parallel_solvers: bool = True  # Executar solvers em paralelo quando BOTH
    max_workers: int = 2  # Máximo workers para paralelização

    # Configurações de logging
    detailed_logging: bool = False  # Log detalhado de cada constraint
    save_results: bool = True  # Salvar resultados em arquivos

    # Configurações experimentais
    compare_solvers: bool = False  # Comparar resultados entre solvers
    generate_counterexamples: bool = True  # Gerar contra-exemplos

    @classmethod
    def from_env(cls) -> "VerificationConfig":
        """Cria configuração a partir de variáveis de ambiente."""

        # Modo de execução
        mode_str = os.getenv("VERIFICATION_MODE", "basic").lower()
        try:
            mode = VerificationMode(mode_str)
        except ValueError:
            logger.warning(f"Invalid mode '{mode_str}', using 'basic'")
            mode = VerificationMode.BASIC

        # Escolha de solver
        solver_str = os.getenv("VERIFICATION_SOLVER", "auto").lower()
        try:
            solver = SolverChoice(solver_str)
        except ValueError:
            logger.warning(f"Invalid solver '{solver_str}', using 'auto'")
            solver = SolverChoice.AUTO

        # Configurações de timeout
        timeout_constraint = float(os.getenv("VERIFICATION_TIMEOUT_CONSTRAINT", "30.0"))
        timeout_total = float(os.getenv("VERIFICATION_TIMEOUT_TOTAL", "300.0"))

        # Configurações de paralelização
        parallel = os.getenv("VERIFICATION_PARALLEL", "true").lower() == "true"
        max_workers = int(os.getenv("VERIFICATION_MAX_WORKERS", "2"))

        # Configurações de logging
        detailed_logging = os.getenv("VERIFICATION_DETAILED_LOG", "false").lower() == "true"
        save_results = os.getenv("VERIFICATION_SAVE_RESULTS", "true").lower() == "true"

        # Configurações experimentais
        compare_solvers = os.getenv("VERIFICATION_COMPARE_SOLVERS", "false").lower() == "true"
        generate_counterexamples = (
            os.getenv("VERIFICATION_GENERATE_COUNTEREXAMPLES", "true").lower() == "true"
        )

        return cls(
            mode=mode,
            solver=solver,
            timeout_per_constraint=timeout_constraint,
            timeout_total=timeout_total,
            parallel_solvers=parallel,
            max_workers=max_workers,
            detailed_logging=detailed_logging,
            save_results=save_results,
            compare_solvers=compare_solvers,
            generate_counterexamples=generate_counterexamples,
        )

    def should_verify(self) -> bool:
        """Retorna se deve executar verificação formal."""
        return self.mode in [VerificationMode.VERIFICATION, VerificationMode.ALL]

    def should_run_inference(self) -> bool:
        """Retorna se deve executar inferência sintética."""
        return self.mode in [VerificationMode.INFERENCE, VerificationMode.ALL]

    def should_run_basic(self) -> bool:
        """Retorna se deve executar algoritmos básicos."""
        return self.mode in [VerificationMode.BASIC, VerificationMode.ALL]

    def get_enabled_solvers(self) -> List[str]:
        """Retorna lista de solvers habilitados.

        Note: AUTO mode now returns BOTH solvers for fair comparison.
        """
        if self.solver == SolverChoice.Z3:
            return ["Z3"]
        elif self.solver == SolverChoice.CVC5:
            return ["CVC5"]
        elif self.solver == SolverChoice.BOTH:
            return ["Z3", "CVC5"]
        else:  # AUTO - usa AMBOS para comparação justa
            return ["Z3", "CVC5"]

    def __str__(self) -> str:
        """Representação string da configuração."""
        return (
            f"VerificationConfig(mode={self.mode.value}, "
            f"solver={self.solver.value}, "
            f"timeout={self.timeout_per_constraint}s)"
        )


# Instância global da configuração
_global_config: Optional[VerificationConfig] = None


def get_verification_config() -> VerificationConfig:
    """Obtém a configuração global de verificação."""
    global _global_config
    if _global_config is None:
        _global_config = VerificationConfig.from_env()
        logger.info(f"🎛️ Configuration loaded: {_global_config}")
    return _global_config


def set_verification_config(config: VerificationConfig):
    """Define a configuração global de verificação."""
    global _global_config
    _global_config = config
    logger.info(f"🎛️ Configuration updated: {config}")


def reload_verification_config():
    """Recarrega a configuração das variáveis de ambiente."""
    global _global_config
    _global_config = None
    return get_verification_config()
