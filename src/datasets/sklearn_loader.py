from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from core.base_dataset import BaseDataset
from utils.types import SklearnDatasetName

class SklearnDatasetLoader(BaseDataset):
    def __init__(self, name: SklearnDatasetName, test_size: float = 0.2, random_state: int = 42):
        self.name = name
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        if self.name == SklearnDatasetName.IRIS:
            data = load_iris()
        elif self.name == SklearnDatasetName.WINE:
            data = load_wine()
        else:
            raise ValueError(f"Dataset '{self.name}' não suportado.")
        
        return train_test_split(
            data.data,
            data.target,
            test_size=self.test_size,
            random_state=self.random_state
        )
