"""Factory for creating dataset loaders based on the dataset source type."""

from datasets.csv_loader import CSVDatasetLoader
from datasets.sklearn_loader import SklearnDatasetLoader
from utils.types import DatasetSourceType

class DatasetFactory:
    """Factory class to instantiate appropriate dataset loader based on source type."""

    _registry = {
        DatasetSourceType.SKLEARN: SklearnDatasetLoader,
        DatasetSourceType.CSV: CSVDatasetLoader,
    }

    @staticmethod
    def register(source_type, loader_cls):
        """Registers a new dataset loader for a given source type.

        Args:
            source_type (DatasetSourceType): The new source type to register.
            loader_cls (Type[BaseDataset]): The loader class.
        """
        DatasetFactory._registry[source_type] = loader_cls

    @staticmethod
    def create(source_type: DatasetSourceType, **kwargs):
        """Instantiates a dataset loader for the given source type.

        Args:
            source_type (DatasetSourceType): The dataset source type (e.g., SKLEARN, CSV).
            **kwargs: Additional arguments required by the specific loader.

        Returns:
            BaseDataset: An instance of the selected dataset loader.

        Raises:
            ValueError: If the provided source_type is unknown.

        Example:
            >>> loader = DatasetFactory.create(DatasetSourceType.SKLEARN, name='iris')
        """
        loader_cls = DatasetFactory._registry.get(source_type)
        if not loader_cls:
            raise ValueError(f"Unknown source type: {source_type}")
        return loader_cls(**kwargs)
