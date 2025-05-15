from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from core.base_dataset import BaseDataset
from utils.types import SklearnDatasetName

class SklearnDatasetLoader(BaseDataset):
    def __init__(self, name: SklearnDatasetName, test_size: float = 0.2, random_state: int = 42):
        self.name = name
        self.test_size = test_size
        self.random_state = random_state
        self.dataset_map = {
            SklearnDatasetName.IRIS: load_iris,
            SklearnDatasetName.WINE: load_wine,
            SklearnDatasetName.BREAST_CANCER: load_breast_cancer,
            SklearnDatasetName.DIGITS: load_digits,
        }

    def load_data(self):
        if self.name not in self.dataset_map:
            raise ValueError(f"Dataset '{self.name}' não suportado.")

        data = self.dataset_map[self.name]()
        return train_test_split(
            data.data, data.target, test_size=self.test_size, random_state=self.random_state
        )

