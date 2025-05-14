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
            "iris": load_iris,
            "wine": load_wine,
            "breast_cancer": load_breast_cancer,
            "digits": load_digits
        }

    def load_data(self):
        name_value = self.name.value if hasattr(self.name, "value") else self.name
        
        if name_value not in self.dataset_map:
            raise ValueError(f"Dataset '{name_value}' não suportado.")
        
        data = self.dataset_map[name_value]()
        return train_test_split(
            data.data, data.target, test_size=self.test_size, random_state=self.random_state
        )

