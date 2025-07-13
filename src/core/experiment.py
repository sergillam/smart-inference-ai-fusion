"""Experiment orchestration class for running models with optional inference and imputation."""

from sklearn.impute import SimpleImputer

class Experiment:
    """Orchestrates the training and evaluation of a model with data imputation.

    Attributes:
        model: Model instance following the framework's interface.
        dataset: Dataset instance used for training and testing.
        inference: Optional inference/perturbation engine.
        impute_strategy (str): Imputation strategy for missing values ('mean', 'median', etc.).
        fill_value: Value used for imputation if strategy is 'constant'.
    """

    def __init__(
        self,
        model,
        dataset,
        *,
        inference=None,
        impute_strategy='mean',
        fill_value=None
    ):
        """Initializes the Experiment with model, dataset, and imputation options.

        Args:
            model: The machine learning model instance.
            dataset: The dataset instance.
            inference (optional): Optional inference or perturbation engine.
            impute_strategy (str, optional): Imputation strategy. Default is 'mean'.
            fill_value (optional): Value for imputation if using 'constant' strategy.
        """
        self.model = model
        self.dataset = dataset
        self.inference = inference
        self.impute_strategy = impute_strategy
        self.fill_value = fill_value

    def run(self, X_train, X_test, y_train, y_test):
        """Runs the experiment: imputes missing data, trains, and evaluates the model.

        Args:
            X_train (array-like): Training features.
            X_test (array-like): Test features.
            y_train (array-like): Training labels.
            y_test (array-like): Test labels.

        Returns:
            dict: Evaluation metrics from the model's `evaluate` method.
        """
        if self.impute_strategy == 'constant':
            imputer = SimpleImputer(strategy='constant', fill_value=self.fill_value or 0)
        else:
            imputer = SimpleImputer(strategy=self.impute_strategy)

        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        self.model.train(X_train, y_train)
        return self.model.evaluate(X_test, y_test)
