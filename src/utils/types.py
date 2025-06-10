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

class DataNoiseConfig(BaseModel):
    noise_level: Optional[float] = None
    truncate_decimals: Optional[int] = None
    quantize_bins: Optional[int] = None
    cast_to_int: Optional[bool] = None
    shuffle_fraction: Optional[float] = None
    scale_range: Optional[Tuple[float, float]] = None
    zero_out_fraction: Optional[float] = None
    insert_nan_fraction: Optional[float] = None
    outlier_fraction: Optional[float] = None
    add_dummy_features: Optional[int] = None
    duplicate_features: Optional[int] = None
    feature_selective_noise: Optional[Tuple[float, List[int]]] = None
    remove_features: Optional[List[int]] = None
    feature_swap: Optional[List[int]] = None
    conditional_noise: Optional[Tuple[int, float, float]] = None
    cluster_swap_fraction: Optional[float] = None
    random_missing_block_fraction: Optional[float] = None
    distribution_shift_fraction: Optional[float] = None
    group_outlier_cluster_fraction: Optional[float] = None
    temporal_drift_std: Optional[float] = None

class LabelNoiseConfig(BaseModel):
    label_noise_fraction: Optional[float] = None
    flip_near_border_fraction: Optional[float] = None
    confusion_matrix_noise_level: Optional[float] = None
    partial_label_fraction: Optional[float] = None
    swap_within_class_fraction: Optional[float] = None
  
class ParameterNoiseConfig(BaseModel):
    integer_noise: Optional[bool] = None
    boolean_flip: Optional[bool] = None
    string_mutator: Optional[bool] = None
    semantic_mutation: Optional[bool] = None
    scale_hyper: Optional[bool] = None
    cross_dependency: Optional[bool] = None
    random_from_space: Optional[bool] = None
    bounded_numeric: Optional[bool] = None
    type_cast_perturbation: Optional[bool] = None
    enum_boundary_shift: Optional[bool] = None

class CSVDatasetName(Enum):
    TITANIC = "datasets/titanic/titanic_dataset.csv"

    @property
    def path(self):
        return self.value
