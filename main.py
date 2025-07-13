"""Main module for running experiments in smart-inference-ai-fusion."""

# from experiments import iris, wine, breast_cancer, digits, titanic
from experiments import digits

def main():
    """Entry point for running selected experiments.

    Uncomment or adjust lines below to select which experiments to run.
    """
    # iris.run_all()
    # wine.run_all()
    # breast_cancer.run_all()
    # digits.run_all()
    # titanic.run_all()
    digits.run_gaussian()
    # digits.run_knn()

if __name__ == "__main__":
    main()
