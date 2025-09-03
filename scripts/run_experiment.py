# File: scripts/run_experiment.py
import argparse
import importlib
import logging
import os
import pkgutil
import runpy
import sys
from datetime import datetime
from types import ModuleType
from typing import List

# Configure console logging only (no file by default)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] --- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def save_error_to_log(module_name: str, error_message: str, full_traceback: str):
    """Save error details to a log file in logs/ directory.
    
    Args:
        module_name (str): Name of the failed module
        error_message (str): Brief error description  
        full_traceback (str): Complete traceback string
    """
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_module_name = module_name.replace(".", "_")
    log_filename = os.path.join(logs_dir, f"error_{safe_module_name}_{timestamp}.log")
    
    # Write error details to file
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write(f"EXPERIMENT ERROR REPORT\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Failed Module: {module_name}\n")
        f.write(f"Error: {error_message}\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"FULL TRACEBACK:\n")
        f.write(f"-" * 30 + "\n")
        f.write(full_traceback)
        f.write(f"\n" + "-" * 30 + "\n")
    
    return log_filename


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


def execute_module(module_name: str) -> tuple[bool, str]:
    """Executes a given module as a script and returns its success status.

    Args:
        module_name (str): The full name of the module to execute.

    Returns:
        tuple[bool, str]: (success_status, error_log_file_path_if_failed)
    """
    try:
        logging.info("▶️  Executing module: %s", module_name)
        runpy.run_module(module_name, run_name="__main__")
        return True, ""
    except Exception as e:
        # Log error to console as before
        logging.error("❌ Failed to execute module: %s", module_name)
        logging.exception(e)
        
        # Capture full traceback for file logging
        import traceback
        full_traceback = traceback.format_exc()
        error_message = str(e)
        
        # Save error details to file
        error_log_file = save_error_to_log(module_name, error_message, full_traceback)
        logging.error("💾 Error details saved to: %s", error_log_file)
        
        return False, error_log_file


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
    failed_experiments = []  # Track which experiments failed
    error_log_files = []  # Track error log files created

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
        success, error_log_file = execute_module(module_name)
        if success:
            succeeded_count += 1
        else:
            failed_count += 1
            failed_experiments.append(module_name)
            if error_log_file:
                error_log_files.append(error_log_file)
            if args.fail_fast:
                logging.error("🔥 --fail-fast enabled. Halting execution.")
                break
    
    # --- Final Report ---
    logging.info("-" * 60)
    logging.info("🏁 Execution Summary:")
    logging.info("   Total experiments executed: %d", total_modules)
    logging.info("   ✅ Succeeded: %d", succeeded_count)
    
    # Enhanced failed experiments report
    if failed_experiments:
        failed_list = ", ".join(failed_experiments)
        logging.info("   ❌ Failed: %d (%s)", failed_count, failed_list)
        
        # Show error log files created
        if error_log_files:
            logging.info("   💾 Error logs saved:")
            for log_file in error_log_files:
                logging.info("      - %s", log_file)
    else:
        logging.info("   ❌ Failed: %d", failed_count)
    
    logging.info("-" * 60)


    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    # sys.exit() ensures the script's exit code is passed to the shell
    sys.exit(main())