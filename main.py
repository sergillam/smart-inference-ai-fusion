#from experiments import iris, wine, breast_cancer, digits, titanic
from experiments import digits

def main():
    # iris.run_all()
    # wine.run_all()
    # breast_cancer.run_all()
    # digits.run_all()
    # titanic.run_all()
    
    digits.run_gaussian()
    #digits.run_knn()
if __name__ == "__main__":
    main()
