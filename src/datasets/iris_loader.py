from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from core.base_dataset import BaseDataset

class IrisLoader(BaseDataset):
    def load_data(self):
        data = load_iris()
        return train_test_split(data.data, data.target, test_size=0.2, random_state=42)
