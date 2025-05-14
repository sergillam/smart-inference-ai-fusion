from enum import Enum

class DatasetSourceType(Enum):
    SKLEARN = "sklearn"
    CSV = "csv"
    # JSON = "json" (futuramente)
    # XLSX = "xlsx"

class ReportMode(Enum):
    """Modos de saída para relatório de dados."""
    PRINT = 'print'
    JSON = 'json'

class SklearnDatasetName(Enum):
    IRIS = "iris"
    WINE = "wine"
    #DIGITS = "digits" (futuramente)