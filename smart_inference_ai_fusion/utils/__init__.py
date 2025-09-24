"""This package contains utility modules for preprocessing, metrics, and logging."""

from .solver_comparison import (
    ComparisonResult,
    SolverComparison,
    SolverMetrics,
    add_comparison_result,
    generate_comparison_report,
    global_comparison,
    print_comparison_summary,
)
from .verification_config import (
    SolverChoice,
    VerificationConfig,
    VerificationMode,
    get_verification_config,
    reload_verification_config,
    set_verification_config,
)

__all__ = [
    "VerificationMode",
    "SolverChoice",
    "VerificationConfig",
    "get_verification_config",
    "set_verification_config",
    "reload_verification_config",
    "SolverMetrics",
    "ComparisonResult",
    "SolverComparison",
    "add_comparison_result",
    "generate_comparison_report",
    "print_comparison_summary",
    "global_comparison",
]
