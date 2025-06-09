from enum import Enum
from typing import List, Tuple, Optional
from pydantic import BaseModel

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
    BREAST_CANCER = "breast_cancer"
    DIGITS = "digits"

class DatasetNoiseConfig(BaseModel):
    noise_level: Optional[float] = None  # Intensidade de ruído gaussiano
    truncate_decimals: Optional[int] = None  # Número de casas decimais
    quantize_bins: Optional[int] = None  # Quantização em N bins
    cast_to_int: Optional[bool] = None  # Converte para int
    shuffle_fraction: Optional[float] = None  # Fração de colunas embaralhadas
    scale_range: Optional[Tuple[float, float]] = None  # Intervalo de escala (min, max)
    zero_out_fraction: Optional[float] = None  # Fração de valores zerados
    insert_nan_fraction: Optional[float] = None  # Fração de NaNs inseridos
    outlier_fraction: Optional[float] = None  # Fração de outliers
    add_dummy_features: Optional[int] = None  # N novas features aleatórias
    duplicate_features: Optional[int] = None  # N features duplicadas
    feature_selective_noise: Optional[Tuple[float, List[int]]] = None  # (nível, índices)
    remove_features: Optional[List[int]] = None  # Índices a remover
    feature_swap: Optional[List[int]] = None  # Índices a trocar entre si
    label_noise_fraction: Optional[float] = None
    flip_near_border_fraction: Optional[float] = None
    confusion_matrix_noise_level: Optional[float] = None
    partial_label_fraction: Optional[float] = None
    swap_within_class_fraction: Optional[float] = None
    conditional_noise: tuple[int, float, float] | None = None  # (feature_index, threshold, std)
    cluster_swap_fraction: Optional[float] = None
    random_missing_block_fraction: Optional[float] = None  # Fração de células a apagar em blocos
    distribution_shift_fraction: Optional[float] = None
    group_outlier_cluster_fraction: Optional[float] = None  # Fração a ser contaminada com outliers em grupo
    temporal_drift_std: Optional[float] = None  # Intensidade do drift temporal progressivo


class CSVDatasetName(Enum):
    TITANIC = "datasets/titanic/titanic_dataset.csv"

    @property
    def path(self):
        return self.value
