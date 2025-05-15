from .knn_titanic import run as run_knn
from .svm_titanic import run as run_svm
from .tree_titanic import run as run_tree
from .perceptron_titanic import run as run_perceptron
from .gaussian_titanic import run as run_gaussian

def run_all():
    print("=== Executando Todos os Experimentos com a Base TITANIC ===")
    run_knn()
    run_svm()
    run_tree()
    run_perceptron()
    run_gaussian()
    print("=== Todos os Experimentos Concluídos ===")