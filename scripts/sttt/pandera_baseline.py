"""Pandera baseline phase for STTT experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from smart_inference_ai_fusion.experiments.common import create_dataset
from smart_inference_ai_fusion.inference.transformations.data.noise import GaussianNoise
from smart_inference_ai_fusion.inference.transformations.label.random_label_noise import RandomLabelNoise
from smart_inference_ai_fusion.inference.transformations.params.scale_hyper import ScaleHyperparameter
from smart_inference_ai_fusion.utils.types import CSVDatasetName, DatasetSourceType

try:
    import pandera.pandas as pa
    from pandera import Check

    HAS_PANDERA = True
except ImportError:  # pragma: no cover - optional dependency in runtime
    pa = None
    Check = None
    HAS_PANDERA = False


@dataclass(frozen=True)
class PanderaPhaseResult:
    wall_clock_ms: float
    caught_fault: bool
    num_failures: int
    failures_by_type: dict[str, int]
    dependency_available: bool


def _data_schema(num_features: int):
    assert HAS_PANDERA and pa is not None and Check is not None
    columns = {
        f"f{i}": pa.Column(
            float,
            checks=[Check.not_nullable(), Check.in_range(0.0, 1.0)],
            nullable=False,
            required=True,
            coerce=True,
        )
        for i in range(num_features)
    }
    return pa.DataFrameSchema(columns=columns, strict=True)


def _label_schema(allowed_labels: set[int | float]):
    assert HAS_PANDERA and pa is not None and Check is not None
    return pa.DataFrameSchema(
        {
            "label": pa.Column(
                int,
                checks=[Check.isin(sorted(allowed_labels))],
                nullable=False,
                required=True,
                coerce=True,
            )
        },
        strict=True,
    )


def _param_schema(params: dict[str, Any], key: str):
    assert HAS_PANDERA and pa is not None and Check is not None
    base_value = params.get(key)
    checks = [Check.not_nullable()]
    if isinstance(base_value, (int, float)):
        checks.append(Check.eq(base_value))
    return pa.DataFrameSchema(
        {
            key: pa.Column(float, checks=checks, nullable=False, required=True, coerce=True),
        },
        strict=True,
    )


def run_pandera_phase(
    *,
    dataset_source: DatasetSourceType,
    dataset_name: CSVDatasetName,
    model_params: dict[str, Any],
    seed: int,
) -> PanderaPhaseResult:
    """Run schema-based validation over data/label/param perturbations."""
    start = time.perf_counter()
    failures_by_type = {"data": 0, "label": 0, "param": 0}

    if not HAS_PANDERA:
        elapsed = (time.perf_counter() - start) * 1000.0
        return PanderaPhaseResult(
            wall_clock_ms=elapsed,
            caught_fault=False,
            num_failures=0,
            failures_by_type=failures_by_type,
            dependency_available=False,
        )

    np.random.seed(seed)
    dataset = create_dataset(dataset_source, dataset_name)
    x_train, x_test, y_train, _ = dataset.load_data()

    x_test_arr = np.asarray(x_test, dtype=np.float64)
    y_train_arr = np.asarray(y_train)

    # Data perturbation: GaussianNoise(sigma=0.01)
    x_data_perturbed = GaussianNoise(level=0.01).apply(x_test_arr)
    data_df = pd.DataFrame(x_data_perturbed, columns=[f"f{i}" for i in range(x_data_perturbed.shape[1])])
    try:
        _data_schema(x_data_perturbed.shape[1]).validate(data_df)
    except Exception:  # pandera error types depend on installed version
        failures_by_type["data"] += 1

    # Label perturbation: RandomLabelNoise(rate=0.05)
    y_label_perturbed = RandomLabelNoise(flip_fraction=0.05).apply(y_train_arr)
    label_df = pd.DataFrame({"label": y_label_perturbed})
    try:
        _label_schema(set(np.unique(y_train_arr).tolist())).validate(label_df)
    except Exception:
        failures_by_type["label"] += 1

    # Parameter perturbation: ScaleHyperparameter(key=max_depth, factors=(0.5, 1.5))
    param_key = "max_depth" if "max_depth" in model_params else "C"
    param_scale = ScaleHyperparameter(key=param_key, factors=(0.5, 1.5))
    perturbed_value = param_scale.apply(dict(model_params))
    param_df = pd.DataFrame([{param_key: perturbed_value}])
    try:
        _param_schema(model_params, param_key).validate(param_df)
    except Exception:
        failures_by_type["param"] += 1

    elapsed = (time.perf_counter() - start) * 1000.0
    num_failures = sum(failures_by_type.values())
    return PanderaPhaseResult(
        wall_clock_ms=elapsed,
        caught_fault=num_failures > 0,
        num_failures=num_failures,
        failures_by_type=failures_by_type,
        dependency_available=True,
    )
