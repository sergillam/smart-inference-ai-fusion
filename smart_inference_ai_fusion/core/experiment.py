"""Experiment orchestration class for running models with optional inference and imputation."""

from typing import Dict

from numpy.typing import ArrayLike
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

    def __init__(self, model, dataset, *, inference=None, impute_strategy="mean", fill_value=None):
        """Initialize the Experiment.

        Args:
            model: The machine learning model instance.
            dataset: The dataset instance.
            inference (optional): Optional inference or perturbation engine.
            impute_strategy (str, optional): Imputation strategy. Defaults to "mean".
            fill_value (optional): Value for imputation when using the "constant" strategy.
        """
        self.model = model
        self.dataset = dataset
        self.inference = inference
        self.impute_strategy = impute_strategy
        self.fill_value = fill_value

    def run(
        self,
        X_train: ArrayLike,
        X_test: ArrayLike,
        y_train: ArrayLike,
        y_test: ArrayLike,
    ) -> Dict:
        """Run the experiment: impute missing data, train, and evaluate the model.

        Args:
            X_train (ArrayLike): Training feature matrix. Shape (n_samples_train, n_features).
            X_test (ArrayLike): Test feature matrix. Shape (n_samples_test, n_features).
            y_train (ArrayLike): Training labels. Shape (n_samples_train,).
            y_test (ArrayLike): Test labels. Shape (n_samples_test,).

        Returns:
            dict: Evaluation metrics produced by the model's ``evaluate`` method.

        Raises:
            ValueError: If the imputation strategy is invalid or data shapes are inconsistent.
        """
        if self.impute_strategy == "constant":
            imputer = SimpleImputer(strategy="constant", fill_value=self.fill_value or 0)
        else:
            imputer = SimpleImputer(strategy=self.impute_strategy)

        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        self.model.train(X_train, y_train)
        return self.model.evaluate(X_test, y_test)
