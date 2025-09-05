"""Main entry point for the experiments package.

This module allows the experiments package to be executed directly with:
    python -m smart_inference_ai_fusion.experiments [dataset_name]

Examples:
    python -m smart_inference_ai_fusion.experiments           # Run all experiments
    python -m smart_inference_ai_fusion.experiments digits    # Run digits experiments only
    python -m smart_inference_ai_fusion.experiments lfw_people # Run LFW People experiments only
"""

import sys

from smart_inference_ai_fusion.utils.logging import setup_logger

from . import run_all_experiments

if __name__ == "__main__":
    # Use the proper logging system that saves to files
    setup_logger("smart_inference_ai_fusion.experiments")

    dataset = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_all_experiments(dataset)
    sys.exit(0 if success else 1)
