# File: scripts/run_experiment.py
import argparse
import importlib
import logging
import os
import pkgutil
import runpy
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import List

# Import custom logging setup (no basicConfig to avoid duplication)
from smart_inference_ai_fusion.utils.logging import logger


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
        logger.info("▶️  Executing module: %s", module_name)
        runpy.run_module(module_name, run_name="__main__")
        return True, ""
    except Exception as e:
        # Log error to console as before
        logger.error("❌ Failed to execute module: %s", module_name)
        logger.exception(e)
        
        # Capture full traceback for file logging
        import traceback
        full_traceback = traceback.format_exc()
        error_message = str(e)
        
        # Save error details to file
        error_log_file = save_error_to_log(module_name, error_message, full_traceback)
        logger.error("💾 Error details saved to: %s", error_log_file)
        
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

    logger.info("📦 '%s' is a package, discovering sub-modules...", module.__name__)
    sub_modules = [
        info.name for info in pkgutil.iter_modules(module.__path__, module.__name__ + '.')
    ]
    
    if not sub_modules:
        logging.warning("🤷 No runnable sub-modules found in '%s'.", module.__name__)
        
    return sorted(sub_modules)


def _is_dataset_package(target: str) -> bool:
    """Check if the target is a dataset package (contains multiple experiments).
    
    Uses dynamic discovery to detect valid experiment packages by checking:
    1. If it's under experiments module
    2. If the target directory exists and contains Python experiment files
    
    Args:
        target: The target module path
        
    Returns:
        bool: True if this is a dataset package with experiments
    """
    parts = target.split('.')
    
    # Must be at least 3 parts: package.experiments.dataset
    if len(parts) < 3 or parts[-2] != 'experiments':
        return False
    
    dataset_part = parts[-1]
    
    try:
        # Try to import as a package to verify it exists and is importable
        import importlib
        target_module = importlib.import_module(target)
        
        # Check if it's actually a package (has __path__)
        if not hasattr(target_module, '__path__'):
            return False
        
        # Check if it contains experiment modules (Python files that are not __init__.py)
        import pkgutil
        submodules = [
            info.name for info in pkgutil.iter_modules(target_module.__path__)
            if not info.name.startswith('__')
        ]
        
        # Must have at least one submodule to be considered a valid dataset package
        return len(submodules) > 0
        
    except (ImportError, AttributeError):
        # If we can't import it or it doesn't have the expected structure, it's not a dataset package
        return False


def _log_enhanced_summary(dataset_name: str, succeeded_count: int, failed_count: int,
                         successful_experiments: List[str], failed_experiments: List[str],
                         error_log_files: List[str]):
    """Log enhanced summary similar to run_all_experiments style.
    
    Args:
        dataset_name: Name of the dataset
        succeeded_count: Number of successful experiments
        failed_count: Number of failed experiments
        successful_experiments: List of successful experiment names
        failed_experiments: List of failed experiment names
        error_log_files: List of error log files created
    """
    logger.info("=" * 70)
    logger.info("🏁 EXECUTION SUMMARY")
    logger.info("=" * 70)
    logger.info("📊 Total experiments executed: %d", succeeded_count + failed_count)
    logger.info("✅ Succeeded: %d", succeeded_count)
    logger.info("❌ Failed: %d", failed_count)
    logger.info("")
    
    if successful_experiments:
        logger.info("🎯 SUCCESSFUL EXPERIMENTS:")
        logger.info("   📂 %s (%d experiments):", dataset_name, len(successful_experiments))
        for exp in successful_experiments:
            logger.info("      ✅ %s", exp)
        logger.info("")
    
    if failed_experiments:
        logger.info("💥 FAILED EXPERIMENTS:")
        logger.info("   📂 %s (%d failures):", dataset_name, len(failed_experiments))
        for exp in failed_experiments:
            logger.info("      ❌ %s", exp)
        logger.info("")
        
        if error_log_files:
            logger.info("💾 Error logs saved:")
            for log_file in error_log_files:
                logger.info("   - %s", log_file)
            logger.info("")
    
    # Show recent result files (similar to original function)
    _log_recent_result_files()
    
    logger.info("=" * 70)


def _log_recent_result_files():
    """Log recent result files from the results directory."""
    try:
        results_dir = Path("results")
        if not results_dir.exists():
            return
            
        # Get recent JSON result files
        json_files = list(results_dir.glob("*.json"))
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if json_files:
            logger.info("📁 RECENT RESULT FILES:")
            for i, file_path in enumerate(json_files[:10], 1):  # Show top 10
                logger.info("    %2d. %s", i, file_path.name)
    except Exception as e:
        logger.debug("Could not list recent result files: %s", e)


def main() -> int:
    """Main execution function for the experiment runner.

    Returns:
        int: An exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()
    
    logger.info("🔎 Importing target: '%s'", args.target)
    try:
        target_module = importlib.import_module(args.target)
    except ImportError as e:
        logger.error("❌ Could not import target '%s': %s", args.target, e)
        return 1

    # Check if this is a dataset package (e.g., smart_inference_ai_fusion.experiments.digits)
    # In this case, use the enhanced run_all_experiments function for better output
    if (_is_dataset_package(args.target)):
        dataset_name = args.target.split('.')[-1]  # Extract dataset name
        logger.info("🎯 Detected dataset package. Using enhanced execution for: %s", dataset_name)
        
        # Execute all experiments for this dataset and capture results
        succeeded_count = 0
        failed_count = 0
        successful_experiments = []
        failed_experiments = []
        error_log_files = []

        # Determine which modules to run
        modules_to_run = discover_submodules(target_module) or [target_module.__name__]

        for module_name in modules_to_run:
            success, error_log_file = execute_module(module_name)
            if success:
                succeeded_count += 1
                # Extract clean experiment name for display
                experiment_name = module_name.split('.')[-1].replace('_wine', '').replace('_digits', '').replace('_', '_')
                successful_experiments.append(f"{dataset_name}.{experiment_name}")
            else:
                failed_count += 1
                experiment_name = module_name.split('.')[-1].replace('_wine', '').replace('_digits', '').replace('_', '_')
                failed_experiments.append(f"{dataset_name}.{experiment_name}")
                if error_log_file:
                    error_log_files.append(error_log_file)
                if args.fail_fast:
                    logger.error("🔥 --fail-fast enabled. Halting execution.")
                    break

        # Enhanced dataset summary in style of run_all_experiments()
        _log_enhanced_summary(dataset_name, succeeded_count, failed_count, 
                            successful_experiments, failed_experiments, error_log_files)
        
        return 0 if failed_count == 0 else 1
    
    # For individual experiments, use the original logic
    succeeded_count = 0
    failed_count = 0
    failed_experiments = []  # Track which experiments failed
    error_log_files = []  # Track error log files created

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
                logger.error("🔥 --fail-fast enabled. Halting execution.")
                break
    
    # --- Final Report ---
    logger.info("-" * 60)
    logger.info("🏁 Execution Summary:")
    logger.info("   Total experiments executed: %d", total_modules)
    logger.info("   ✅ Succeeded: %d", succeeded_count)
    
    # Enhanced failed experiments report
    if failed_experiments:
        failed_list = ", ".join(failed_experiments)
        logger.info("   ❌ Failed: %d (%s)", failed_count, failed_list)
        
        # Show error log files created
        if error_log_files:
            logger.info("   💾 Error logs saved:")
            for log_file in error_log_files:
                logger.info("      - %s", log_file)
    else:
        logger.info("   ❌ Failed: %d", failed_count)
    
    logger.info("-" * 60)


    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    # sys.exit() ensures the script's exit code is passed to the shell
    sys.exit(main())