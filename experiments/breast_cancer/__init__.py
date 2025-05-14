from .knn_breast_cancer import run as run_knn
from .svm_breast_cancer import run as run_svm
from .tree_iris import run as run_tree
from .perceptron_breast_cancer import run as run_perceptron
from .gaussian_breast_cancer import run as run_gaussian

def run_all():
    print("=== Executando Todos os Experimentos com a Base Breast Cancer ===")
    run_knn()
    run_svm()
    run_tree()
    run_perceptron()
    run_gaussian()
    print("=== Todos os Experimentos Concluídos ===")