"""Main entry point for the experiments package.

This module allows the experiments package to be executed directly with:
    python -m smart_inference_ai_fusion.experiments [dataset_name] [options]

Examples:
    python -m smart_inference_ai_fusion.experiments           # Run all experiments
    python -m smart_inference_ai_fusion.experiments digits    # Run digits experiments only
    python -m smart_inference_ai_fusion.experiments wine --mode all --solvers z3,cvc5
    python -m smart_inference_ai_fusion.experiments wine --mode verification --solvers z3
"""

import argparse
import os
import sys

from smart_inference_ai_fusion.utils.logging import setup_logger
from smart_inference_ai_fusion.verification.core.error_handling import (
    set_circuit_breaker,
    reset_error_handler,
)

from . import run_all_experiments


def parse_arguments():
    """Parse command line arguments for experiment execution."""
    parser = argparse.ArgumentParser(
        description="Execute Smart Inference AI Fusion experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m smart_inference_ai_fusion.experiments wine
  python -m smart_inference_ai_fusion.experiments wine --mode verification
  python -m smart_inference_ai_fusion.experiments wine --mode verification --solvers z3
  python -m smart_inference_ai_fusion.experiments wine --mode all --solvers z3,cvc5
  python -m smart_inference_ai_fusion.experiments digits --debug
        """,
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name (wine, digits, etc.) - if not provided, runs all datasets",
    )

    parser.add_argument(
        "--mode",
        choices=["basic", "inference", "verification", "all"],
        default="all",
        help="Execution mode (default: all)",
    )

    parser.add_argument(
        "--solvers", help="Comma-separated list of solvers: z3, cvc5 (e.g., 'z3,cvc5' or 'z3')"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument("--parallel", action="store_true", help="Enable parallel solver execution")

    return parser.parse_args()


def setup_environment_variables(args):
    """Configure environment variables based on command line arguments."""
    # Set verification mode
    os.environ["VERIFICATION_MODE"] = args.mode

    # Configure solvers
    if args.solvers:
        solvers_list = [s.strip() for s in args.solvers.split(",")]
        if len(solvers_list) == 1:
            os.environ["VERIFICATION_SOLVER"] = solvers_list[0]
        else:
            os.environ["VERIFICATION_SOLVER"] = "both"
            os.environ["VERIFICATION_PARALLEL"] = "true"
            os.environ["VERIFICATION_COMPARE_SOLVERS"] = "true"
    else:
        os.environ["VERIFICATION_SOLVER"] = "auto"

    # Configure parallel execution
    if args.parallel:
        os.environ["VERIFICATION_PARALLEL"] = "true"

    # Configure logging level
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["VERIFICATION_DETAILED_LOG"] = "true"

    # Enable verification by default for verification modes
    if args.mode in ["verification", "all"]:
        os.environ["VERIFICATION_ENABLED"] = "true"


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Setup environment variables
    setup_environment_variables(args)

    # Use the proper logging system that saves to files
    setup_logger("smart_inference_ai_fusion.experiments")

    # IMPORTANTE: Desabilitar circuit breaker para experimentos científicos
    # Isso garante que AMBOS os solvers (Z3 e CVC5) recebam os mesmos dados
    set_circuit_breaker(False)
    reset_error_handler()

    # Execute experiments
    success = run_all_experiments(args.dataset)
    sys.exit(0 if success else 1)
