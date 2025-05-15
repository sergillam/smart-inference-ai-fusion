import pandas as pd
from sklearn.model_selection import train_test_split
from core.base_dataset import BaseDataset
from utils.types import CSVDatasetName
from utils.preprocessing import preprocess_titanic

class CSVDatasetLoader(BaseDataset):
    def __init__(self, file_path: CSVDatasetName, target_column, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessing_map = {
            CSVDatasetName.TITANIC: preprocess_titanic,
        }

    def load_data(self):
        df = pd.read_csv(self.file_path.value)

        preprocess_fn = self.preprocessing_map.get(self.file_path)
        # Aplica pré-processamento se houver
        if preprocess_fn:
            df = preprocess_fn(df)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)


