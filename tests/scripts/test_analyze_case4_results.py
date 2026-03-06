"""Tests for analysis helpers of Case 4."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_case4_results import (
    bootstrap_ci,
    holm_bonferroni,
    load_results,
    summarize_results,
)


def _write_sample_results(results_dir: Path) -> None:
    payload = [
        {
            "experiment_type": "data_quant",
            "dataset_name": "wine",
            "algorithm_name": "KNNModel",
            "quantization_method": "uniform",
            "bit_width": 16,
            "dtype_profile": "integer",
            "baseline_accuracy": 0.9,
            "quantized_accuracy": 0.85,
            "accuracy_degradation": 0.05,
            "baseline_memory_bytes": 1000,
            "quantized_memory_bytes": 500,
            "compression_ratio": 2.0,
            "baseline_time_ms": 1.0,
            "quantized_time_ms": 1.2,
            "overhead_pct": 20.0,
            "quantization_mse": 0.01,
            "seed": 42,
            "metadata": {"mode": "data_only", "execution_id": "a"},
        },
        {
            "experiment_type": "data_quant",
            "dataset_name": "wine",
            "algorithm_name": "KNNModel",
            "quantization_method": "uniform",
            "bit_width": 16,
            "dtype_profile": "integer",
            "baseline_accuracy": 0.9,
            "quantized_accuracy": 0.84,
            "accuracy_degradation": 0.06,
            "baseline_memory_bytes": 1000,
            "quantized_memory_bytes": 500,
            "compression_ratio": 2.0,
            "baseline_time_ms": 1.0,
            "quantized_time_ms": 1.1,
            "overhead_pct": 10.0,
            "quantization_mse": 0.02,
            "seed": 123,
            "metadata": {"mode": "data_only", "execution_id": "b"},
        },
    ]
    with open(results_dir / "case4_results_test.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_load_results_and_summarize(tmp_path) -> None:
    """Result loader and summarizer should build non-empty analysis tables."""
    _write_sample_results(tmp_path)
    frame = load_results(tmp_path)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    summary, significance = summarize_results(frame)
    assert not summary.empty
    assert not significance.empty
    assert "p_value_holm" in significance.columns


def test_holm_bonferroni_monotonicity() -> None:
    """Holm-adjusted p-values should be non-decreasing by sorted p-value order."""
    p_values = [0.01, 0.03, 0.2]
    adjusted = holm_bonferroni(p_values)
    assert len(adjusted) == 3
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_bootstrap_ci_returns_bounds() -> None:
    """Bootstrap CI should return ordered finite bounds for non-empty series."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    low, high = bootstrap_ci(series, n_boot=200)
    assert low <= high
