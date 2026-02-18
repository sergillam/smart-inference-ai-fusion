"""Sistema robusto de tratamento de erros e fallback para verificação multi-solver."""

import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Níveis de severidade de erro."""

    LOW = "low"  # Erro recuperável, continuar com fallback
    MEDIUM = "medium"  # Erro significativo, tentar fallback limitado
    HIGH = "high"  # Erro crítico, parar execução
    CRITICAL = "critical"  # Erro fatal do sistema


class FallbackStrategy(Enum):
    """Estratégias de fallback."""

    NONE = "none"  # Não fazer fallback
    NEXT_SOLVER = "next_solver"  # Tentar próximo solver
    DEFAULT_SOLVER = "default_solver"  # Usar solver padrão (Z3)
    SEQUENTIAL = "sequential"  # Tentar todos os solvers sequencialmente
    GRACEFUL_DEGRADATION = "graceful"  # Reduzir funcionalidade mas continuar


@dataclass
class ErrorContext:
    """Contexto de um erro para debugging e recuperação."""

    error_type: str
    error_message: str
    solver_name: str
    operation: str
    timestamp: float
    stack_trace: str
    severity: ErrorSeverity
    recovery_attempted: bool = False
    recovery_successful: bool = False
    fallback_strategy: Optional[FallbackStrategy] = None


# Flag global para desabilitar circuit breaker (para experimentos científicos)
# pylint: disable=invalid-name
_circuit_breaker_enabled = True


def set_circuit_breaker(enabled: bool):
    """Habilita/desabilita o circuit breaker para experimentos científicos.

    Quando desabilitado, solvers nunca serão desabilitados automaticamente,
    garantindo que AMBOS os solvers (Z3 e CVC5) recebam os mesmos dados.
    """
    global _circuit_breaker_enabled
    _circuit_breaker_enabled = enabled
    status = "ENABLED" if enabled else "DISABLED"
    logger.info(f"⚡ Circuit breaker {status}")


def is_circuit_breaker_enabled() -> bool:
    """Retorna se o circuit breaker está habilitado."""
    return _circuit_breaker_enabled


class VerificationErrorHandler:
    """Manipulador robusto de erros para verificação formal."""

    def __init__(self, fallback_strategy: FallbackStrategy = FallbackStrategy.NEXT_SOLVER):
        self.fallback_strategy = fallback_strategy
        self.error_history: List[ErrorContext] = []
        self.solver_reliability: Dict[str, float] = {}
        self.recovery_strategies: Dict[str, Callable] = {
            "timeout": self._handle_timeout_error,
            "memory": self._handle_memory_error,
            "syntax": self._handle_syntax_error,
            "solver_unavailable": self._handle_solver_unavailable,
            "constraint_unsupported": self._handle_unsupported_constraint,
            "installation": self._handle_installation_error,
            "import": self._handle_import_error,
        }

        logger.info("🛡️ Sistema de error handling inicializado")

    def handle_verification_error(
        self, error: Exception, solver_name: str, operation: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Manipula erro de verificação com estratégia de recuperação."""
        context = context or {}

        # Classificar erro e criar contexto
        error_context = self._classify_error(error, solver_name, operation)
        self.error_history.append(error_context)

        # Atualizar confiabilidade do solver
        self._update_solver_reliability(solver_name, False)

        # Log detalhado do erro
        self._log_error(error_context, context)

        # Tentar recuperação baseada no tipo de erro
        recovery_result = self._attempt_recovery(error_context, context)

        # Aplicar estratégia de fallback se necessário
        if not recovery_result.get("success", False):
            fallback_result = self._apply_fallback_strategy(error_context, context)
            recovery_result.update(fallback_result)

        return recovery_result

    def _classify_error(self, error: Exception, solver_name: str, operation: str) -> ErrorContext:
        """Classifica o erro e determina severidade."""
        error_message = str(error)
        error_type = type(error).__name__

        # Determinação de severidade baseada no tipo de erro
        if "timeout" in error_message.lower() or "time limit" in error_message.lower():
            severity = ErrorSeverity.LOW
            error_category = "timeout"
        elif "memory" in error_message.lower() or "out of memory" in error_message.lower():
            severity = ErrorSeverity.MEDIUM
            error_category = "memory"
        elif "import" in error_message.lower() or "module" in error_message.lower():
            severity = ErrorSeverity.HIGH
            error_category = "import"
        elif "install" in error_message.lower() or "not found" in error_message.lower():
            severity = ErrorSeverity.HIGH
            error_category = "installation"
        elif (
            "unrecognized option" in error_message.lower() or "unsupported" in error_message.lower()
        ):
            severity = ErrorSeverity.MEDIUM
            error_category = "constraint_unsupported"
        elif "syntax" in error_message.lower() or "parse" in error_message.lower():
            severity = ErrorSeverity.MEDIUM
            error_category = "syntax"
        else:
            severity = ErrorSeverity.MEDIUM
            error_category = "unknown"

        return ErrorContext(
            error_type=error_category,
            error_message=error_message,
            solver_name=solver_name,
            operation=operation,
            timestamp=time.time(),
            stack_trace=traceback.format_exc(),
            severity=severity,
        )

    def _attempt_recovery(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Tenta recuperação específica baseada no tipo de erro."""
        recovery_strategy = self.recovery_strategies.get(error_context.error_type)

        if not recovery_strategy:
            logger.warning(f"⚠️ No recovery strategy for error: {error_context.error_type}")
            return {"success": False, "message": "No recovery strategy available"}

        try:
            error_context.recovery_attempted = True
            result = recovery_strategy(error_context, context)
            error_context.recovery_successful = result.get("success", False)
            return result
        except Exception as recovery_error:
            logger.error(f"❌ Recovery failed: {recovery_error}")
            return {"success": False, "message": f"Recovery failed: {recovery_error}"}

    def _handle_timeout_error(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recupera de erro de timeout."""
        logger.info(f"⏰ Recuperando de timeout em {error_context.solver_name}")

        # Estratégias para timeout:
        # 1. Reduzir timeout e tentar novamente
        # 2. Simplificar constraints
        # 3. Usar solver mais rápido

        return {
            "success": True,
            "message": "Timeout handled, trying with reduced constraints",
            "action": "reduce_timeout",
            "new_timeout": context.get("timeout", 30) * 0.5,
            "simplified_constraints": True,
        }

    def _handle_memory_error(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recupera de erro de memória."""
        logger.info(f"💾 Recovering from memory error in {error_context.solver_name}")

        return {
            "success": True,
            "message": "Memory error handled, reducing problem complexity",
            "action": "reduce_complexity",
            "batch_size": min(context.get("batch_size", 100), 10),
            "simplified_constraints": True,
        }

    def _handle_syntax_error(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recupera de erro de sintaxe."""
        logger.info(f"📝 Recuperando de erro de sintaxe em {error_context.solver_name}")

        return {
            "success": True,
            "message": "Syntax error handled, using basic constraints",
            "action": "use_basic_constraints",
            "fallback_to_basic": True,
        }

    def _handle_solver_unavailable(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recupera quando solver não está disponível."""
        logger.warning(f"🚫 Solver {error_context.solver_name} not available")

        return {
            "success": False,  # Precisa de fallback para outro solver
            "message": f"Solver {error_context.solver_name} unavailable",
            "action": "switch_solver",
            "requires_fallback": True,
        }

    def _handle_unsupported_constraint(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recupera de constraint não suportado."""
        logger.info(f"🔧 Constraint not supported in {error_context.solver_name}")

        return {
            "success": True,
            "message": "Unsupported constraint handled, using compatible subset",
            "action": "filter_constraints",
            "compatible_constraints_only": True,
        }

    def _handle_installation_error(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recupera de erro de instalação."""
        logger.error(f"📦 Installation error in {error_context.solver_name}")

        return {
            "success": False,
            "message": f"Installation error for {error_context.solver_name}",
            "action": "disable_solver",
            "requires_fallback": True,
            "suggestion": f"Run: pip install {error_context.solver_name.lower()}",
        }

    def _handle_import_error(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recupera de erro de importação."""
        logger.error(f"📥 Import error in {error_context.solver_name}")

        return {
            "success": False,
            "message": f"Import error for {error_context.solver_name}",
            "action": "disable_solver",
            "requires_fallback": True,
            "suggestion": f"Check {error_context.solver_name} installation",
        }

    def _apply_fallback_strategy(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aplica estratégia de fallback configurada."""
        error_context.fallback_strategy = self.fallback_strategy

        if self.fallback_strategy == FallbackStrategy.NONE:
            return {"success": False, "message": "No fallback configured"}

        elif self.fallback_strategy == FallbackStrategy.NEXT_SOLVER:
            return self._fallback_to_next_solver(error_context, context)

        elif self.fallback_strategy == FallbackStrategy.DEFAULT_SOLVER:
            return self._fallback_to_default_solver(error_context, context)

        elif self.fallback_strategy == FallbackStrategy.SEQUENTIAL:
            return self._fallback_sequential(error_context, context)

        elif self.fallback_strategy == FallbackStrategy.GRACEFUL_DEGRADATION:
            return self._fallback_graceful_degradation(error_context, context)

        return {"success": False, "message": "Unknown fallback strategy"}

    def _fallback_to_next_solver(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback para próximo solver disponível."""
        available_solvers = context.get("available_solvers", ["Z3", "CVC5"])
        current_solver = error_context.solver_name

        try:
            current_index = available_solvers.index(current_solver)
            next_solvers = available_solvers[current_index + 1 :]
        except ValueError:
            next_solvers = [s for s in available_solvers if s != current_solver]

        if next_solvers:
            next_solver = next_solvers[0]
            logger.info(f"🔄 Fallback: {current_solver} → {next_solver}")
            return {
                "success": True,
                "message": f"Falling back to {next_solver}",
                "action": "switch_solver",
                "new_solver": next_solver,
            }

        return {"success": False, "message": "No alternative solver available"}

    def _fallback_to_default_solver(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback para solver padrão (Z3)."""
        if error_context.solver_name == "Z3":
            return {"success": False, "message": "Default solver already failed"}

        logger.info(f"🏠 Fallback to default solver: Z3")
        return {
            "success": True,
            "message": "Falling back to default solver (Z3)",
            "action": "switch_solver",
            "new_solver": "Z3",
        }

    def _fallback_sequential(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Tenta todos os solvers sequencialmente."""
        available_solvers = context.get("available_solvers", ["Z3", "CVC5"])
        failed_solver = error_context.solver_name

        remaining_solvers = [s for s in available_solvers if s != failed_solver]

        if remaining_solvers:
            logger.info(f"🔄 Fallback sequencial: tentando {remaining_solvers}")
            return {
                "success": True,
                "message": "Trying remaining solvers sequentially",
                "action": "try_all_solvers",
                "remaining_solvers": remaining_solvers,
            }

        return {"success": False, "message": "All solvers exhausted"}

    def _fallback_graceful_degradation(
        self, error_context: ErrorContext, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Degradação graciosa - reduz funcionalidade mas continua."""
        logger.info(f"⬇️ Graceful degradation for {error_context.solver_name}")

        return {
            "success": True,
            "message": "Graceful degradation - reduced functionality",
            "action": "reduce_functionality",
            "verification_disabled": True,
            "basic_mode_only": True,
            "warning": "Verification disabled due to errors",
        }

    def _update_solver_reliability(self, solver_name: str, success: bool):
        """Atualiza métricas de confiabilidade do solver."""
        if solver_name not in self.solver_reliability:
            self.solver_reliability[solver_name] = 1.0

        # Atualização exponencial da confiabilidade
        current = self.solver_reliability[solver_name]
        if success:
            self.solver_reliability[solver_name] = min(1.0, current + 0.1)
        else:
            self.solver_reliability[solver_name] = max(0.0, current - 0.2)

    def _log_error(self, error_context: ErrorContext, context: Dict[str, Any]):
        """Log detalhado do erro para debugging."""
        severity_emoji = {
            ErrorSeverity.LOW: "🟡",
            ErrorSeverity.MEDIUM: "🟠",
            ErrorSeverity.HIGH: "🔴",
            ErrorSeverity.CRITICAL: "💀",
        }

        emoji = severity_emoji.get(error_context.severity, "❓")

        logger.error(
            f"{emoji} ERRO {error_context.severity.value.upper()}: {error_context.solver_name}"
        )
        logger.error(f"   Operation: {error_context.operation}")
        logger.error(f"   Tipo: {error_context.error_type}")
        logger.error(f"   Mensagem: {error_context.error_message}")

        if context:
            logger.debug(f"   Contexto: {context}")

        if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.debug(f"   Stack trace: {error_context.stack_trace}")

    def get_error_summary(self) -> Dict[str, Any]:
        """Retorna resumo de erros para análise."""
        if not self.error_history:
            return {"total_errors": 0}

        # Análise por solver
        errors_by_solver = {}
        errors_by_type = {}
        errors_by_severity = {}

        for error in self.error_history:
            # Por solver
            solver = error.solver_name
            if solver not in errors_by_solver:
                errors_by_solver[solver] = {"count": 0, "recovery_rate": 0}
            errors_by_solver[solver]["count"] += 1
            if error.recovery_successful:
                errors_by_solver[solver]["recovery_rate"] += 1

        # Calcular taxas de recuperação
        for solver_data in errors_by_solver.values():
            if solver_data["count"] > 0:
                solver_data["recovery_rate"] = solver_data["recovery_rate"] / solver_data["count"]

        # Por tipo
        for error in self.error_history:
            error_type = error.error_type
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1

        # Por severidade
        for error in self.error_history:
            severity = error.severity.value
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "errors_by_solver": errors_by_solver,
            "errors_by_type": errors_by_type,
            "errors_by_severity": errors_by_severity,
            "solver_reliability": dict(self.solver_reliability),
            "recovery_success_rate": sum(1 for e in self.error_history if e.recovery_successful)
            / len(self.error_history),
        }

    def should_disable_solver(self, solver_name: str) -> bool:
        """Determina se um solver deve ser desabilitado devido a muitos erros.

        NOTA: Se o circuit breaker estiver desabilitado (para experimentos científicos),
        esta função SEMPRE retorna False, garantindo que ambos os solvers recebam
        os mesmos dados para verificação.
        """
        # Se circuit breaker desabilitado, nunca desabilitar solvers
        if not _circuit_breaker_enabled:
            return False

        reliability = self.solver_reliability.get(solver_name, 1.0)
        recent_errors = sum(1 for e in self.error_history[-10:] if e.solver_name == solver_name)

        return reliability < 0.3 or recent_errors > 5

    def reset(self):
        """Reseta o estado do error handler (útil entre experimentos)."""
        self.error_history.clear()
        self.solver_reliability.clear()
        logger.info("🔄 Error handler resetado")


# Instância global do manipulador de erros
global_error_handler = VerificationErrorHandler()


def handle_verification_error(
    error: Exception, solver_name: str, operation: str, context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Função de conveniência para manipulação de erros."""
    return global_error_handler.handle_verification_error(error, solver_name, operation, context)


def get_error_summary() -> Dict[str, Any]:
    """Função de conveniência para obter resumo de erros."""
    return global_error_handler.get_error_summary()


def should_disable_solver(solver_name: str) -> bool:
    """Função de conveniência para verificar se solver deve ser desabilitado."""
    return global_error_handler.should_disable_solver(solver_name)


def reset_error_handler():
    """Função de conveniência para resetar o error handler."""
    global_error_handler.reset()
