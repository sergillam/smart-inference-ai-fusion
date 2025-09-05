"""Main entry point for the experiments package.

This module allows the experiments package to be executed directly with:
    python -m smart_inference_ai_fusion.experiments [dataset_name]

Examples:
    python -m smart_inference_ai_fusion.experiments           # Run all experiments
    python -m smart_inference_ai_fusion.experiments digits    # Run digits experiments only
    python -m smart_inference_ai_fusion.experiments lfw_people # Run LFW People experiments only
"""

import logging
import sys

from . import run_all_experiments

if __name__ == "__main__":
    # Configure logging to match single experiment runs
    logging.basicConfig(level=logging.INFO, format="INFO: %(message)s", force=True)

    dataset = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_all_experiments(dataset)
    sys.exit(0 if success else 1)
