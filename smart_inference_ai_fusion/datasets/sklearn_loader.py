"""Loader for standard scikit-learn datasets."""

from typing import Tuple

from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split

from smart_inference_ai_fusion.core.base_dataset import BaseDataset
from smart_inference_ai_fusion.utils.types import SklearnDatasetName


class SklearnDatasetLoader(BaseDataset):
    """Loader for scikit-learn datasets with train/test split interface.

    Attributes:
        name (SklearnDatasetName): Dataset identifier (Enum).
        test_size (float): Fraction for test split (default 0.2).
        random_state (int): Seed for reproducibility.
        dataset_map (dict): Maps dataset names to scikit-learn loader functions.
    """

    def __init__(self, name: SklearnDatasetName, test_size: float = 0.2, random_state: int = 42):
        """Initializes the dataset loader.

        Args:
            name (SklearnDatasetName): Which dataset to load (Enum value).
            test_size (float, optional): Fraction for test split. Defaults to 0.2.
            random_state (int, optional): Seed for reproducibility. Defaults to 42.
        """
        self.name = name
        self.test_size = test_size
        self.random_state = random_state
        self.dataset_map = {
            SklearnDatasetName.IRIS: load_iris,
            SklearnDatasetName.WINE: load_wine,
            SklearnDatasetName.BREAST_CANCER: load_breast_cancer,
            SklearnDatasetName.DIGITS: load_digits,
        }

    def load_data(self) -> Tuple:
        """Loads the dataset and returns the train/test split.

        Returns:
            Tuple: (X_train, X_test, y_train, y_test), all as numpy arrays.

        Raises:
            ValueError: If the given dataset name is not supported.
        """
        if self.name not in self.dataset_map:
            raise ValueError(f"Dataset '{self.name}' not supported.")

        data = self.dataset_map[self.name]()
        return train_test_split(
            data.data, data.target, test_size=self.test_size, random_state=self.random_state
        )
