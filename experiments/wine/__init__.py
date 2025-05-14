from .knn_wine import run as run_knn
from .svm_wine import run as run_svm
from .tree_wine import run as run_tree
from .perceptron_wine import run as run_perceptron
from .gaussian_wine import run as run_gaussian

def run_all():
    print("=== Executando Todos os Experimentos com a Base WINE ===")
    run_knn()
    run_svm()
    run_tree()
    run_perceptron()
    run_gaussian()
    print("=== Todos os Experimentos Concluídos ===")