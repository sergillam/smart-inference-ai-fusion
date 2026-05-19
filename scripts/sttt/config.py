"""Static configuration for STTT experiment runs."""

from __future__ import annotations

from dataclasses import dataclass

from smart_inference_ai_fusion.models.logistic_regression_model import LogisticRegressionModel
from smart_inference_ai_fusion.models.mlp_model import MLPModel
from smart_inference_ai_fusion.models.random_forest_classifier_model import RandomForestClassifierModel
from smart_inference_ai_fusion.models.tree_model import DecisionTreeModel
from smart_inference_ai_fusion.utils.types import (
    CSVDatasetName,
    DatasetSourceType,
    SklearnDatasetName,
)

SEEDS = list(range(1, 31))


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    source: DatasetSourceType
    name: CSVDatasetName | SklearnDatasetName


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_class: type
    params: dict


DATASETS: dict[str, DatasetSpec] = {
    "wine": DatasetSpec("wine", DatasetSourceType.SKLEARN, SklearnDatasetName.WINE),
    "wids": DatasetSpec("wids", DatasetSourceType.CSV, CSVDatasetName.WIDS_ICU),
    "ieee": DatasetSpec("ieee", DatasetSourceType.CSV, CSVDatasetName.IEEE_FRAUD),
}

MODELS: dict[str, ModelSpec] = {
    "lr": ModelSpec(
        "lr",
        LogisticRegressionModel,
        {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    ),
    "dt": ModelSpec("dt", DecisionTreeModel, {"criterion": "gini", "max_depth": 5}),
    "rf": ModelSpec(
        "rf",
        RandomForestClassifierModel,
        {"n_estimators": 50, "max_depth": 5, "bootstrap": True},
    ),
    "mlp": ModelSpec(
        "mlp",
        MLPModel,
        {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 500,
        },
    ),
}
