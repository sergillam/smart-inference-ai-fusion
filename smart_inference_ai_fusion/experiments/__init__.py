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
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


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

    if not experiments:
        logger.warning("No experiments found for dataset: %s", dataset)
        return 0, 0

    for experiment_module in experiments:
        if _run_single_experiment(experiment_module):
            success_count += 1
        else:
            failed_count += 1

    return success_count, failed_count


def run_all_experiments(dataset_name: Optional[str] = None) -> bool:
    """Run all experiments for a dataset or for all discovered datasets.

    This function now orchestrates the discovery of datasets and delegates the
    actual execution to helper functions, reducing its complexity.

    Args:
        dataset_name: Specific dataset to run. If None, auto-discovers all datasets.

    Returns:
        bool: True if all experiments succeeded, False if any failed.
    """
    if dataset_name:
        datasets = [dataset_name]
    else:
        experiments_path = Path(__file__).parent
        datasets = sorted(
            [
                item.name
                for item in experiments_path.iterdir()
                if item.is_dir() and not item.name.startswith("_")
            ]
        )

    total_success = 0
    total_failed = 0
    failed_experiments = []

    for dataset in datasets:
        success, failed = _run_experiments_for_dataset(dataset)
        total_success += success
        total_failed += failed
        if failed > 0:
            failed_experiments.append(dataset)

    # Final Summary matching run_experiment.py format
    logger.info("-" * 60)
    logger.info("🏁 Execution Summary:")
    logger.info("   Total experiments executed: %d", total_success + total_failed)
    logger.info("   ✅ Succeeded: %d", total_success)

    if failed_experiments:
        failed_list = ", ".join(failed_experiments)
        logger.info("   ❌ Failed: %d (%s)", total_failed, failed_list)
    else:
        logger.info("   ❌ Failed: %d", total_failed)

    logger.info("-" * 60)

    return total_failed == 0


def run():
    """Legacy entry point for backwards compatibility."""
    return run_all_experiments()


if __name__ == "__main__":
    # Configure basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    selected_dataset = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_all_experiments(selected_dataset)
    sys.exit(0 if success else 1)
