from abc import ABC, abstractmethod

class ParameterTransformation(ABC):
    """Base class for parameter inference transformations."""
    @abstractmethod
    def apply(self, params: dict) -> dict:
        pass