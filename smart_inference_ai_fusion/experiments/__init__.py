"""Experiments package - Auto-discovery and execution of ML experiments.

This package provides automated discovery and execution of experiments across
different datasets. Supports both individual experiment execution and batch
runs with comprehensive logging and error handling.

Usage:
    # Run all experiments in a dataset
    python -m smart_inference_ai_fusion.experiments digits

    # Run specific experiment
    python -m smart_inference_ai_fusion.experiments.digits.gaussian_digits

    # From code
    from smart_inference_ai_fusion.experiments import run_all_experiments
    run_all_experiments("digits")
"""

import importlib
import logging
import pkgutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Global variables to track experiment results across datasets
_successful_experiments: List[str] = []
_failed_experiments: List[str] = []


def _initialize_experiment_tracking():
    """Initialize global experiment tracking variables."""
    _successful_experiments.clear()
    _failed_experiments.clear()


def _discover_datasets() -> List[str]:
    """Discover all available dataset directories.

    Returns:
        List of dataset directory names.
    """
    experiments_path = Path(__file__).parent
    return sorted(
        [
            item.name
            for item in experiments_path.iterdir()
            if item.is_dir() and not item.name.startswith("_")
        ]
    )


def _group_experiments_by_dataset(experiments: List[str]) -> dict:
    """Group experiment names by dataset.

    Args:
        experiments: List of experiment names in format 'dataset.model'.

    Returns:
        Dictionary mapping dataset names to lists of model names.
    """
    by_dataset = {}
    for exp in experiments:
        dataset, model = exp.split(".", 1)
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(model)
    return by_dataset


def _log_successful_experiments():
    """Log detailed information about successful experiments."""
    if not _successful_experiments:
        return

    logger.info("")
    logger.info("🎯 SUCCESSFUL EXPERIMENTS:")
    by_dataset = _group_experiments_by_dataset(_successful_experiments)

    for dataset, models in sorted(by_dataset.items()):
        logger.info("   📂 %s (%d experiments):", dataset, len(models))
        for model in sorted(models):
            logger.info("      ✅ %s.%s", dataset, model)


def _log_failed_experiments():
    """Log detailed information about failed experiments."""
    if not _failed_experiments:
        return

    logger.info("")
    logger.info("💥 FAILED EXPERIMENTS:")
    by_dataset = _group_experiments_by_dataset(_failed_experiments)

    for dataset, models in sorted(by_dataset.items()):
        logger.info("   📂 %s (%d failures):", dataset, len(models))
        for model in sorted(models):
            logger.info("      ❌ %s.%s", dataset, model)


def _log_recent_result_files():
    """Log information about recently generated result files."""
    logger.info("")
    logger.info("📁 RECENT RESULT FILES:")

    try:
        results_dir = Path("results")
        if not results_dir.exists():
            logger.info("   (Results directory not found)")
            return

        # Get files modified in the last hour
        recent_files = []
        one_hour_ago = datetime.now() - timedelta(hours=1)

        for file_path in results_dir.glob("*.json"):
            if file_path.stat().st_mtime > one_hour_ago.timestamp():
                recent_files.append(file_path.name)

        if recent_files:
            recent_files.sort()
            for i, filename in enumerate(recent_files[-10:], 1):  # Show last 10
                logger.info("   %2d. %s", i, filename)
        else:
            logger.info("   (No recent result files found)")

    except (OSError, IOError) as e:
        logger.info("   (Could not list result files: %s)", str(e))


def discover_dataset_experiments(dataset_name: str) -> List[str]:
    """Auto-discover all experiments for a given dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'digits', 'wine', 'lfw_people')

    Returns:
        List of discovered experiment module names
    """
    experiment_modules = []
    dataset_path = Path(__file__).parent / dataset_name

    if not dataset_path.exists():
        logger.warning("Dataset path not found: %s", dataset_path)
        return []

    # Discover all Python modules in the dataset directory
    try:
        dataset_package = f"smart_inference_ai_fusion.experiments.{dataset_name}"
        package = importlib.import_module(dataset_package)

        for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            if not module_name.endswith(".__init__"):
                experiment_modules.append(module_name)

    except ImportError as err:
        logger.error("Failed to import dataset package %s: %s", dataset_name, err)

    return sorted(experiment_modules)


def _run_single_experiment(experiment_module: str) -> bool:
    """Imports and runs a single experiment module.

    This helper function encapsulates the logic for importing, finding the 'run'
    function, and executing it with detailed error handling.

    Args:
        experiment_module: The full name of the module to run.

    Returns:
        True if the experiment ran successfully, False otherwise.
    """
    logger.info("▶️  Executing: %s", experiment_module)
    try:
        module = importlib.import_module(experiment_module)
    except ImportError as err:
        logger.error("❌ Failed to import %s: %s", experiment_module, err)
        return False

    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        logger.warning("No callable 'run' function found in %s", experiment_module)
        return False

    try:
        run_fn()
    except (ValueError, RuntimeError) as err:
        logger.error("❌ Expected failure in %s: %s", experiment_module, err)
        return False
    except Exception as err:  # pylint: disable=broad-exception-caught
        logger.error("❌ Unexpected failure in %s: %s", experiment_module, err, exc_info=True)
        return False

    logger.info("✅ Completed: %s", experiment_module)
    return True


def _run_experiments_for_dataset(dataset: str) -> Tuple[int, int]:
    """Runs all experiments for a single specified dataset.

    Args:
        dataset: The name of the dataset.

    Returns:
        A tuple containing the count of successful and failed experiments.
    """
    logger.info("🔬 Running experiments for dataset: %s", dataset)
    experiments = discover_dataset_experiments(dataset)
    success_count, failed_count = 0, 0

    # Track detailed results for summary
    successful_experiments = []
    failed_experiments = []

    if not experiments:
        logger.warning("No experiments found for dataset: %s", dataset)
        return 0, 0

    for experiment_module in experiments:
        # Extract model name from module (e.g., "mlp_newsgroups_20" -> "mlp")
        model_name = experiment_module.split(".")[-1].replace(f"_{dataset}", "")

        if _run_single_experiment(experiment_module):
            success_count += 1
            successful_experiments.append(f"{dataset}.{model_name}")
        else:
            failed_count += 1
            failed_experiments.append(f"{dataset}.{model_name}")

    # Store results for final summary
    _successful_experiments.extend(successful_experiments)
    _failed_experiments.extend(failed_experiments)

    return success_count, failed_count


def run_all_experiments(dataset_name: Optional[str] = None) -> bool:
    """Run all experiments for a dataset or for all discovered datasets.

    This function orchestrates the discovery of datasets and delegates the
    actual execution to helper functions, keeping complexity low.

    Args:
        dataset_name: Specific dataset to run. If None, auto-discovers all datasets.

    Returns:
        bool: True if all experiments succeeded, False if any failed.
    """
    _initialize_experiment_tracking()

    datasets = [dataset_name] if dataset_name else _discover_datasets()

    total_success, total_failed = _execute_all_datasets(datasets)

    _log_execution_summary(total_success, total_failed)

    return total_failed == 0


def _execute_all_datasets(datasets: List[str]) -> Tuple[int, int]:
    """Execute experiments for all specified datasets.

    Args:
        datasets: List of dataset names to process.

    Returns:
        Tuple of (total_success_count, total_failed_count).
    """
    total_success = 0
    total_failed = 0

    for dataset in datasets:
        success, failed = _run_experiments_for_dataset(dataset)
        total_success += success
        total_failed += failed

    return total_success, total_failed


def _log_execution_summary(total_success: int, total_failed: int):
    """Log the complete execution summary with detailed results.

    Args:
        total_success: Total number of successful experiments.
        total_failed: Total number of failed experiments.
    """
    logger.info("=" * 70)
    logger.info("🏁 EXECUTION SUMMARY")
    logger.info("=" * 70)
    logger.info("📊 Total experiments executed: %d", total_success + total_failed)
    logger.info("✅ Succeeded: %d", total_success)
    logger.info("❌ Failed: %d", total_failed)

    _log_successful_experiments()
    _log_failed_experiments()
    _log_recent_result_files()

    logger.info("=" * 70)


def run():
    """Legacy entry point for backwards compatibility."""
    return run_all_experiments()


if __name__ == "__main__":
    # Configure basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    selected_dataset = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_all_experiments(selected_dataset)
    sys.exit(0 if success else 1)
