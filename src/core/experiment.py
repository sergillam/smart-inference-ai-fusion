from sklearn.impute import SimpleImputer

class Experiment:
    def __init__(
        self, 
        model, 
        dataset, 
        inference=None, 
        impute_strategy='mean',
        fill_value=None  # Apenas usado se strategy == 'constant'
    ):
        self.model = model
        self.dataset = dataset
        self.inference = inference
        self.impute_strategy = impute_strategy
        self.fill_value = fill_value

    def run(self, X_train, X_test, y_train, y_test):
        if self.impute_strategy == 'constant':
            imputer = SimpleImputer(strategy='constant', fill_value=self.fill_value or 0)
        else:
            imputer = SimpleImputer(strategy=self.impute_strategy)

        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        self.model.train(X_train, y_train)
        return self.model.evaluate(X_test, y_test)
