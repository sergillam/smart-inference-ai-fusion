"""Main entry point for running experiments."""

from smart_inference_ai_fusion.experiments import digits


def main():
    """Run all experiments."""
    print("=== Running All Experiments ===")
    digits.run_all()


if __name__ == "__main__":
    main()
