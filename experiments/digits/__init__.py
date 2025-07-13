"""Digits experiments package.

This package organizes and runs experiments for the Digits dataset.
"""

# from .knn_digits import run as run_knn
# from .svm_digits import run as run_svm
# from .tree_wine import run as run_tree
# from .perceptron_digits import run as run_perceptron
from .gaussian_digits import run as run_gaussian

def run_all():
    """Run all configured experiments for the Digits dataset."""
    print("=== Executando Todos os Experimentos com a Base DIGITS ===")
    # run_knn()
    # run_svm()
    # run_tree()
    # run_perceptron()
    run_gaussian()
    print("=== Todos os Experimentos Concluídos ===")
