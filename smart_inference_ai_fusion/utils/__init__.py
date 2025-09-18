"""This package contains utility modules for preprocessing, metrics, and logging."""

from .verification_config import (
    VerificationMode,
    SolverChoice,
    VerificationConfig,
    get_verification_config,
    set_verification_config,
    reload_verification_config
)

from .solver_comparison import (
    SolverMetrics,
    ComparisonResult,
    SolverComparison,
    add_comparison_result,
    generate_comparison_report,
    print_comparison_summary,
    global_comparison
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
    "global_comparison"
]
