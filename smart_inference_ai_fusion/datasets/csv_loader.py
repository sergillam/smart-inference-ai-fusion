"""CSVDatasetLoader for custom CSV datasets with preprocessing."""

import pandas as pd
from sklearn.model_selection import train_test_split

from smart_inference_ai_fusion.core.base_dataset import BaseDataset
from smart_inference_ai_fusion.utils.preprocessing import preprocess_titanic
from smart_inference_ai_fusion.utils.types import CSVDatasetName


class CSVDatasetLoader(BaseDataset):
    """Loader for custom CSV datasets with preprocessing.

    This class supports loading and preprocessing of custom CSV datasets, integrating
    with the base dataset interface used in the framework.

    Attributes:
        file_path (CSVDatasetName): Path or enum for the CSV file.
        target_column (str): Name of the target column.
        test_size (float): Fraction of the data to reserve for testing.
        random_state (int): Seed for random train/test split.
        preprocessing_map (dict): Maps dataset enums to preprocessing functions.
    """

    def __init__(
        self,
        file_path: CSVDatasetName,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initializes the CSVDatasetLoader.

        Args:
            file_path (CSVDatasetName): Enum or string representing the CSV dataset path.
            target_column (str): Name of the target column for prediction.
            test_size (float, optional): Test set proportion. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessing_map = {
            CSVDatasetName.TITANIC: preprocess_titanic,
        }

    def load_data(self):
        """Loads and preprocesses the dataset, then splits into train and test sets.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) after preprocessing and splitting.
        """
        df = pd.read_csv(self.file_path.value)

        preprocess_fn = self.preprocessing_map.get(self.file_path)
        if preprocess_fn:
            df = preprocess_fn(df)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
