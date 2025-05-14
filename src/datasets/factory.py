from datasets.csv_loader import CSVDatasetLoader
from datasets.sklearn_loader import SklearnDatasetLoader
from utils.types import DatasetSourceType

class DatasetFactory:
    @staticmethod
    def create(source_type: DatasetSourceType, **kwargs):
        if source_type == DatasetSourceType.SKLEARN:
            return SklearnDatasetLoader(**kwargs)
        elif source_type == DatasetSourceType.CSV:
            return CSVDatasetLoader(**kwargs)
        else:
            raise ValueError(f"Tipo de origem desconhecido: {source_type}")
