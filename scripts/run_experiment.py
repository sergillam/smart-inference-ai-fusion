# File: scripts/run_experiment.py
import argparse
import importlib
import logging
import pkgutil
import runpy
import sys
from types import ModuleType
from typing import List

# Settings logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] --- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_arguments() -> argparse.Namespace:
    """Configures and parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Runs one or more experiment modules.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "target",
        help="The full import path of the experiment module or package to run (e.g., 'smart_inference_ai_fusion.experiments.digits').",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop execution immediately if any experiment fails.",
    )
    return parser.parse_args()


def execute_module(module_name: str) -> bool:
    """Executes a given module as a script and returns its success status.

    Args:
        module_name (str): The full name of the module to execute.

    Returns:
        bool: True if the execution was successful, False otherwise.
    """
    try:
        logging.info("▶️  Executing module: %s", module_name)
        runpy.run_module(module_name, run_name="__main__")
        return True
    except Exception as e:
        logging.error("❌ Failed to execute module: %s", module_name)
        # logging.exception provides the full traceback for better debugging
        logging.exception(e)
        return False


def discover_submodules(module: ModuleType) -> List[str]:
    """Discovers all runnable sub-modules within a given package.

    Args:
        module (ModuleType): The imported package module.

    Returns:
        List[str]: A sorted list of submodule names.
    """
    if not hasattr(module, "__path__"):
        return []

    logging.info("📦 '%s' is a package, discovering sub-modules...", module.__name__)
    sub_modules = [
        info.name for info in pkgutil.iter_modules(module.__path__, module.__name__ + '.')
    ]
    
    if not sub_modules:
        logging.warning("🤷 No runnable sub-modules found in '%s'.", module.__name__)
        
    return sorted(sub_modules)


def main() -> int:
    """Main execution function for the experiment runner.

    Returns:
        int: An exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()
    succeeded_count = 0
    failed_count = 0

    logging.info("🔎 Importing target: '%s'", args.target)
    try:
        target_module = importlib.import_module(args.target)
    except ImportError as e:
        logging.error("❌ Could not import target '%s': %s", args.target, e)
        return 1

    # Determine which modules to run
    modules_to_run = discover_submodules(target_module) or [target_module.__name__]
    total_modules = len(modules_to_run)

    for module_name in modules_to_run:
        if execute_module(module_name):
            succeeded_count += 1
        else:
            failed_count += 1
            if args.fail_fast:
                logging.error("🔥 --fail-fast enabled. Halting execution.")
                break
    
    # --- Final Report ---
    print("-" * 60)
    logging.info("🏁 Execution Summary:")
    logging.info("   Total experiments executed: %d", total_modules)
    logging.info("   ✅ Succeeded: %d", succeeded_count)
    logging.info("   ❌ Failed: %d", failed_count)
    print("-" * 60)

    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    # sys.exit() ensures the script's exit code is passed to the shell
    sys.exit(main())