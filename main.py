from experiments import iris, wine, breast_cancer, digits, titanic

def main():
    iris.run_all()
    wine.run_all()
    breast_cancer.run_all()
    digits.run_all()
    titanic.run_all()

if __name__ == "__main__":
    main()
