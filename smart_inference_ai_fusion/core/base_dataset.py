"""Abstract base class for all dataset loaders."""

from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Abstract base class for dataset loaders.

    Any custom dataset loader should inherit from this class and
    implement the `load_data` method.
    """

    @abstractmethod
    def load_data(self):
        """Loads and returns the dataset split into train and test sets.

        Returns:
            Typically returns a tuple of (X_train, X_test, y_train, y_test),
            but concrete implementations may define their own return signature.
        """
