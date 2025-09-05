"""Categorize type the data used enumeration for the inference framework."""

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel


class DatasetSourceType(Enum):
    """Enumerates supported dataset sources.

    Attributes:
        SKLEARN: Built-in scikit-learn datasets.
        CSV: Custom datasets in CSV format.
    """

    SKLEARN = "sklearn"
    CSV = "csv"
    # JSON = "json" (future)
    # XLSX = "xlsx"


class ReportMode(Enum):
    """Specifies output/reporting modes for experiments.

    Attributes:
        PRINT: Outputs to logger/console.
        JSON_LOG: Saves to the logs/ folder.
        JSON_RESULT: Saves to the results/ folder.
    """

    PRINT = "print"
    JSON_LOG = "json_log"
    JSON_RESULT = "json_result"


class SklearnDatasetName(Enum):
    """Lists supported scikit-learn dataset names.

    Attributes:
        IRIS: Iris classification dataset.
        WINE: Wine classification dataset.
        BREAST_CANCER: Breast cancer dataset.
        DIGITS: Handwritten digits dataset.
        LFW_PEOPLE: Labeled Faces in the Wild people dataset.
        MAKE_MOONS: Synthetic make_moons dataset for clustering.
        NEWSGROUPS_20: 20 Newsgroups text classification dataset.
    """

    IRIS = "iris"
    WINE = "wine"
    BREAST_CANCER = "breast_cancer"
    DIGITS = "digits"
    LFW_PEOPLE = "lfw_people"
    MAKE_MOONS = "make_moons"
    NEWSGROUPS_20 = "newsgroups_20"


class DataNoiseConfig(BaseModel):
    """Configuration for applying synthetic noise or perturbations to data (features X).

    Attributes:
        noise_level (Optional[float]): Global Gaussian noise level.
        truncate_decimals (Optional[int]): Number of decimals to keep (truncation).
        quantize_bins (Optional[int]): Number of bins for quantization.
        cast_to_int (Optional[bool]): Whether to cast features to int.
        shuffle_fraction (Optional[float]): Fraction of features to shuffle.
        scale_range (Optional[Tuple[float, float]]): Range for feature scaling.
        zero_out_fraction (Optional[float]): Fraction of features to zero out.
        insert_nan_fraction (Optional[float]): Fraction of features to replace with NaN.
        outlier_fraction (Optional[float]): Fraction of features to inject as outliers.
        add_dummy_features (Optional[int]): Number of dummy features to add.
        duplicate_features (Optional[int]): Number of features to duplicate.
        feature_selective_noise (Optional[Tuple[float, List[int]]]): Noise level and
            indices for selective noise.
        remove_features (Optional[List[int]]): List of feature indices to remove.
        feature_swap (Optional[List[int]]): Indices of features to swap.
        conditional_noise (Optional[Tuple[int, float, float]]): Conditional noise
            (feature idx, threshold, std).
        cluster_swap_fraction (Optional[float]): Fraction for cluster swapping.
        random_missing_block_fraction (Optional[float]): Fraction for block missing data.
        distribution_shift_fraction (Optional[float]): Fraction for distribution shift.
        group_outlier_cluster_fraction (Optional[float]): Fraction for group outlier injection.
        temporal_drift_std (Optional[float]): Standard deviation for temporal drift.
    """

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
    """Configuration for label inference/noise techniques.

    Attributes:
        label_noise_fraction (Optional[float]): Random label noise fraction.
        flip_near_border_fraction (Optional[float]): Fraction of labels to flip near
            model decision border.
        confusion_matrix_noise_level (Optional[float]): Noise based on confusion matrix structure.
        partial_label_fraction (Optional[float]): Fraction for partial label/noise techniques.
        swap_within_class_fraction (Optional[float]): Fraction for swapping labels within
            the same class.
    """

    label_noise_fraction: Optional[float] = None
    flip_near_border_fraction: Optional[float] = None
    confusion_matrix_noise_level: Optional[float] = None
    partial_label_fraction: Optional[float] = None
    swap_within_class_fraction: Optional[float] = None


class ParameterNoiseConfig(BaseModel):
    """Configuration for parameter (hyperparameter) inference techniques.

    Attributes:
        integer_noise (Optional[bool]): Apply integer parameter perturbation.
        boolean_flip (Optional[bool]): Apply boolean parameter flipping.
        string_mutator (Optional[bool]): Mutate string-based parameters.
        semantic_mutation (Optional[bool]): Apply semantic-level replacements
            (e.g., "gini" <-> "entropy").
        scale_hyper (Optional[bool]): Scale numeric hyperparameters.
        cross_dependency (Optional[bool]): Perturb parameters based on dependencies.
        random_from_space (Optional[bool]): Randomly select from valid parameter space.
        bounded_numeric (Optional[bool]): Apply bounded shift to numeric hyperparameters.
        type_cast_perturbation (Optional[bool]): Apply type casting to hyperparameters.
        enum_boundary_shift (Optional[bool]): Apply perturbation at enum boundaries.
    """

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
    """Supported CSV dataset filenames.

    Attributes:
        TITANIC: Titanic CSV dataset file.
    """

    TITANIC = "datasets/titanic/titanic_dataset.csv"

    @property
    def path(self) -> str:
        """Return the path to the CSV dataset file.

        Returns:
            str: Absolute or relative path to the dataset CSV file.
        """
        return self.value
