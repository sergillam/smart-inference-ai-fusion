"""Loader for standard scikit-learn datasets."""

from typing import Tuple

from sklearn.datasets import (
    fetch_20newsgroups_vectorized,
    fetch_lfw_people,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_moons,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
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
            SklearnDatasetName.LFW_PEOPLE: self._load_lfw_people,
            SklearnDatasetName.MAKE_MOONS: self._load_make_moons,
            SklearnDatasetName.NEWSGROUPS_20: self._load_newsgroups_20,
        }

    def _load_newsgroups_20(self):
        """Load the 20 Newsgroups dataset with dimensionality reduction to avoid OOM.

        Returns:
            Bunch object with data and target attributes.
        """
        data = fetch_20newsgroups_vectorized(subset="all")

        # Apply feature selection first to reduce memory usage
        # Select top 5000 features using chi2
        selector = SelectKBest(chi2, k=min(5000, data.data.shape[1]))
        x_selected = selector.fit_transform(data.data, data.target)

        # Apply TruncatedSVD for further dimensionality reduction
        # Reduce to 200 components (manageable for clustering)
        svd = TruncatedSVD(n_components=min(200, x_selected.shape[1]), random_state=42)
        x_reduced = svd.fit_transform(x_selected)

        return type("Bunch", (), {"data": x_reduced, "target": data.target})()

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

    def _load_lfw_people(self):
        """Load LFW People dataset with optimized settings for faster experiments.

        Returns:
            Bunch object with data and target attributes.
        """
        # Use smaller subset for faster experiments - min_faces_per_person=50 reduces to ~5 people
        # resize=0.3 further reduces image size for speed
        return fetch_lfw_people(min_faces_per_person=50, resize=0.3)

    def _load_make_moons(self):
        """Load synthetic make_moons dataset.

        Returns:
            Bunch-like object with data and target attributes.
        """
        # Create synthetic dataset with noise
        X, y = make_moons(n_samples=1000, noise=0.3, random_state=self.random_state)

        # Return in scikit-learn Bunch format
        class MockBunch:
            def __init__(self, data, target):
                self.data = data
                self.target = target

        return MockBunch(X, y)
