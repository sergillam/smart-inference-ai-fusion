"""Main entry point for running experiments."""

from smart_inference_ai_fusion.experiments import digits
from smart_inference_ai_fusion.utils.report import ReportMode, report_data


def main():
    """Run all experiments."""
    report_data("=== Running All Experiments ===", mode=ReportMode.PRINT)
    digits.run_all()


if __name__ == "__main__":
    main()
