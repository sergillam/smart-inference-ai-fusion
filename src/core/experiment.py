from sklearn.impute import SimpleImputer

class Experiment:
    def __init__(self, model, dataset, inference=None):
        self.model = model
        self.dataset = dataset
        self.inference = inference

    def run(self):
        X_train, X_test, y_train, y_test = self.dataset.load_data()

        if self.inference:
            X_train, X_test = self.inference.apply(X_train, X_test)

        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        self.model.train(X_train, y_train)
        return self.model.evaluate(X_test, y_test)
