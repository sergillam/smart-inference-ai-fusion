"""SIP-Q quantization experiment orchestration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

import numpy as np

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.datasets.factory import DatasetFactory
from smart_inference_ai_fusion.quantization.core.config import QuantizationConfig
from smart_inference_ai_fusion.quantization.core.result import QuantizationResult
from smart_inference_ai_fusion.quantization.data.feature_quantizer import FeatureQuantizer
from smart_inference_ai_fusion.quantization.evaluation.benchmark import (
    benchmark_inference,
    compute_compression_ratio,
    compute_overhead_pct,
    estimate_memory_bytes,
)
from smart_inference_ai_fusion.quantization.evaluation.metrics import (
    compute_clustering_metrics,
    compute_quantization_mse,
    compute_supervised_metrics,
)
from smart_inference_ai_fusion.quantization.model.weight_quantizer import WeightQuantizer
from smart_inference_ai_fusion.utils.types import (
    CSVDatasetName,
    DatasetSourceType,
    SklearnDatasetName,
)

CSV_DATASET_TARGET_COLUMNS = {CSVDatasetName.TITANIC: "Survived"}


class QuantizationExperiment:
    """Run baseline and quantized experiment modes for one dataset/model combination."""

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def run_supervised(
        self,
        dataset_source: DatasetSourceType,
        dataset_name: SklearnDatasetName | CSVDatasetName,
        model_class: type[BaseModel],
        model_params: dict[str, Any],
        *,
        seed: int = 42,
        skip_execution_ids: set[str] | None = None,
    ) -> list[QuantizationResult]:
        """Run baseline + quantization modes for supervised tasks."""
        mode_plan = self._build_mode_plan(skip_execution_ids, dataset_name, model_class, seed)
        if not mode_plan:
            return []
        x_train, x_test, y_train, y_test = self._load_dataset(dataset_source, dataset_name, seed)

        baseline_model = self._build_model(model_class, model_params)
        self._fit_model(baseline_model, x_train, y_train)
        y_pred_baseline = np.asarray(baseline_model.predict(x_test))
        baseline_time_ms = benchmark_inference(baseline_model.predict, x_test)
        baseline_model_bytes = self._estimate_model_memory_bytes(baseline_model)

        baseline_sup = compute_supervised_metrics(y_test, y_pred_baseline, y_pred_baseline)
        results: list[QuantizationResult] = []
        for mode, bit_width, execution_id in mode_plan:
            result = self._run_supervised_mode(
                mode=mode,
                bit_width=bit_width,
                execution_id=execution_id,
                dataset_name=dataset_name,
                model_class=model_class,
                model_params=model_params,
                baseline_model=baseline_model,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                y_pred_baseline=y_pred_baseline,
                baseline_accuracy=baseline_sup["baseline_accuracy"],
                baseline_time_ms=baseline_time_ms,
                baseline_model_bytes=baseline_model_bytes,
                seed=seed,
            )
            results.append(result)
        return results

    def run_unsupervised(
        self,
        dataset_source: DatasetSourceType,
        dataset_name: SklearnDatasetName | CSVDatasetName,
        model_class: type[BaseModel],
        model_params: dict[str, Any],
        *,
        seed: int = 42,
        skip_execution_ids: set[str] | None = None,
    ) -> list[QuantizationResult]:
        """Run baseline + quantization modes for unsupervised tasks."""
        mode_plan = self._build_mode_plan(skip_execution_ids, dataset_name, model_class, seed)
        if not mode_plan:
            return []
        x_train, x_test, y_train, y_test = self._load_dataset(dataset_source, dataset_name, seed)

        baseline_model = self._build_model(model_class, model_params)
        self._fit_model(baseline_model, x_train, y_train)
        labels_baseline = np.asarray(baseline_model.predict(x_test))
        baseline_time_ms = benchmark_inference(baseline_model.predict, x_test)
        baseline_model_bytes = self._estimate_model_memory_bytes(baseline_model)

        baseline_cluster = compute_clustering_metrics(
            x_test, labels_baseline, labels_baseline, y_test
        )
        results: list[QuantizationResult] = []
        for mode, bit_width, execution_id in mode_plan:
            result = self._run_unsupervised_mode(
                mode=mode,
                bit_width=bit_width,
                execution_id=execution_id,
                dataset_name=dataset_name,
                model_class=model_class,
                model_params=model_params,
                baseline_model=baseline_model,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                labels_baseline=labels_baseline,
                baseline_silhouette=baseline_cluster["baseline_silhouette"],
                baseline_time_ms=baseline_time_ms,
                baseline_model_bytes=baseline_model_bytes,
                seed=seed,
            )
            results.append(result)
        return results

    def _run_supervised_mode(
        self,
        *,
        mode: str,
        bit_width: int,
        execution_id: str,
        dataset_name: SklearnDatasetName | CSVDatasetName,
        model_class: type[BaseModel],
        model_params: dict[str, Any],
        baseline_model: BaseModel,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_pred_baseline: np.ndarray,
        baseline_accuracy: float,
        baseline_time_ms: float,
        baseline_model_bytes: int,
        seed: int,
    ) -> QuantizationResult:
        common = self._prepare_quantization_mode(
            mode=mode,
            bit_width=bit_width,
            model_class=model_class,
            model_params=model_params,
            baseline_model=baseline_model,
            baseline_model_bytes=baseline_model_bytes,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
        )

        y_pred_quantized = np.asarray(common["quantized_model"].predict(common["x_test_input"]))
        metrics = compute_supervised_metrics(y_test, y_pred_baseline, y_pred_quantized)
        quantized_time_ms = benchmark_inference(
            common["quantized_model"].predict, common["x_test_input"]
        )
        overhead_pct = compute_overhead_pct(baseline_time_ms, quantized_time_ms)

        baseline_memory_bytes, quantized_memory_bytes = self._compute_mode_memory_bytes(
            mode=mode,
            data_bytes_base=common["data_bytes_base"],
            data_bytes_quant=common["data_bytes_quant"],
            model_bytes_base=common["model_bytes_base"],
            model_bytes_quant=common["model_bytes_quant"],
        )
        compression_ratio = (
            baseline_memory_bytes / quantized_memory_bytes
            if quantized_memory_bytes > 0
            else compute_compression_ratio(64, bit_width)
        )

        return QuantizationResult(
            experiment_type=self._to_result_experiment_type(mode),
            dataset_name=self._dataset_label(dataset_name),
            algorithm_name=model_class.__name__,
            quantization_method=self.config.method,
            bit_width=bit_width,
            dtype_profile=self.config.dtype_profile,
            baseline_accuracy=baseline_accuracy,
            quantized_accuracy=metrics["quantized_accuracy"],
            accuracy_degradation=metrics["accuracy_degradation"],
            baseline_memory_bytes=int(baseline_memory_bytes),
            quantized_memory_bytes=int(quantized_memory_bytes),
            compression_ratio=float(compression_ratio),
            baseline_time_ms=float(baseline_time_ms),
            quantized_time_ms=float(quantized_time_ms),
            overhead_pct=float(overhead_pct),
            quantization_mse=float(
                (common["data_mse"] + common["model_mse"]) / (2 if mode == "hybrid" else 1)
            ),
            seed=seed,
            metadata=self._build_metadata(mode, execution_id, bit_width),
        )

    def _run_unsupervised_mode(
        self,
        *,
        mode: str,
        bit_width: int,
        execution_id: str,
        dataset_name: SklearnDatasetName | CSVDatasetName,
        model_class: type[BaseModel],
        model_params: dict[str, Any],
        baseline_model: BaseModel,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        labels_baseline: np.ndarray,
        baseline_silhouette: float | None,
        baseline_time_ms: float,
        baseline_model_bytes: int,
        seed: int,
    ) -> QuantizationResult:
        common = self._prepare_quantization_mode(
            mode=mode,
            bit_width=bit_width,
            model_class=model_class,
            model_params=model_params,
            baseline_model=baseline_model,
            baseline_model_bytes=baseline_model_bytes,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
        )

        labels_quantized = np.asarray(common["quantized_model"].predict(common["x_test_input"]))
        cluster_metrics = compute_clustering_metrics(
            x_test, labels_baseline, labels_quantized, y_test
        )
        quantized_time_ms = benchmark_inference(
            common["quantized_model"].predict, common["x_test_input"]
        )
        overhead_pct = compute_overhead_pct(baseline_time_ms, quantized_time_ms)

        baseline_memory_bytes, quantized_memory_bytes = self._compute_mode_memory_bytes(
            mode=mode,
            data_bytes_base=common["data_bytes_base"],
            data_bytes_quant=common["data_bytes_quant"],
            model_bytes_base=common["model_bytes_base"],
            model_bytes_quant=common["model_bytes_quant"],
        )
        compression_ratio = (
            baseline_memory_bytes / quantized_memory_bytes
            if quantized_memory_bytes > 0
            else compute_compression_ratio(64, bit_width)
        )

        return QuantizationResult(
            experiment_type=self._to_result_experiment_type(mode),
            dataset_name=self._dataset_label(dataset_name),
            algorithm_name=model_class.__name__,
            quantization_method=self.config.method,
            bit_width=bit_width,
            dtype_profile=self.config.dtype_profile,
            baseline_silhouette=self._coalesce_optional_float(baseline_silhouette),
            quantized_silhouette=self._coalesce_optional_float(
                cluster_metrics["quantized_silhouette"]
            ),
            silhouette_degradation=self._coalesce_optional_float(
                cluster_metrics["silhouette_degradation"]
            ),
            baseline_memory_bytes=int(baseline_memory_bytes),
            quantized_memory_bytes=int(quantized_memory_bytes),
            compression_ratio=float(compression_ratio),
            baseline_time_ms=float(baseline_time_ms),
            quantized_time_ms=float(quantized_time_ms),
            overhead_pct=float(overhead_pct),
            quantization_mse=float(
                (common["data_mse"] + common["model_mse"]) / (2 if mode == "hybrid" else 1)
            ),
            seed=seed,
            metadata=self._build_metadata(mode, execution_id, bit_width),
        )

    def _prepare_quantization_mode(
        self,
        *,
        mode: str,
        bit_width: int,
        model_class: type[BaseModel],
        model_params: dict[str, Any],
        baseline_model: BaseModel,
        baseline_model_bytes: int,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
    ) -> dict[str, Any]:
        x_test_input = x_test
        model_bytes_base = baseline_model_bytes
        model_bytes_quant = baseline_model_bytes
        data_bytes_base = estimate_memory_bytes(x_test)
        data_bytes_quant = estimate_memory_bytes(x_test)
        data_mse = 0.0
        model_mse = 0.0

        if mode in {"data_only", "hybrid"}:
            fq = FeatureQuantizer(
                method=self.config.method,
                num_bits=bit_width,
                dtype_profile=self.config.dtype_profile,
            )
            x_train_q = fq.fit_transform(x_train)
            x_test_q = fq.transform(x_test)
            x_train_use = fq.inverse_transform(x_train_q)
            x_test_use = fq.inverse_transform(x_test_q)
            data_bytes_quant = estimate_memory_bytes(x_test_q)
            data_mse = compute_quantization_mse(x_test, x_test_use)
            x_test_input = x_test_use
        else:
            x_train_use = x_train

        if mode == "model_only":
            model_run = baseline_model
        else:
            model_run = self._build_model(model_class, model_params)
            self._fit_model(model_run, x_train_use, y_train)

        if mode in {"model_only", "hybrid"}:
            wq = WeightQuantizer(
                num_bits=bit_width,
                method=self.config.method,
                dtype_profile=self.config.dtype_profile,
            )
            quantized_model = wq.quantize_model(model_run)
            model_bytes_quant = self._estimate_quantized_model_memory_bytes(model_run, bit_width)
            model_mse = self._compute_model_quantization_mse(model_run, quantized_model)
        else:
            quantized_model = model_run

        return {
            "quantized_model": quantized_model,
            "x_test_input": x_test_input,
            "model_bytes_base": model_bytes_base,
            "model_bytes_quant": model_bytes_quant,
            "data_bytes_base": data_bytes_base,
            "data_bytes_quant": data_bytes_quant,
            "data_mse": data_mse,
            "model_mse": model_mse,
        }

    @staticmethod
    def _compute_mode_memory_bytes(
        *,
        mode: str,
        data_bytes_base: int,
        data_bytes_quant: int,
        model_bytes_base: int,
        model_bytes_quant: int,
    ) -> tuple[int, int]:
        baseline_memory_bytes = (
            data_bytes_base if mode == "data_only" else data_bytes_base + model_bytes_base
        )
        quantized_memory_bytes = (
            data_bytes_quant
            if mode == "data_only"
            else (
                data_bytes_base + model_bytes_quant
                if mode == "model_only"
                else data_bytes_quant + model_bytes_quant
            )
        )
        return int(baseline_memory_bytes), int(quantized_memory_bytes)

    def _build_mode_plan(
        self,
        skip_execution_ids: set[str] | None,
        dataset_name: SklearnDatasetName | CSVDatasetName,
        model_class: type[BaseModel],
        seed: int,
    ) -> list[tuple[str, int, str]]:
        skip = skip_execution_ids or set()
        tasks: list[tuple[str, int, str]] = []
        data_bits = set(self.config.data_bits)
        model_bits = set(self.config.model_bits)
        all_bits = sorted(data_bits | model_bits)

        for bit in all_bits:
            if self._skip_bit_for_dtype_profile(bit):
                continue
            for mode in self._modes_for_bit(bit, data_bits, model_bits):
                execution_id = self._build_execution_id(
                    dataset_name=dataset_name,
                    algorithm_name=model_class.__name__,
                    mode=mode,
                    bit_width=bit,
                    seed=seed,
                )
                if execution_id in skip:
                    continue
                tasks.append((mode, bit, execution_id))
        return tasks

    def _modes_for_bit(self, bit: int, data_bits: set[int], model_bits: set[int]) -> list[str]:
        modes: list[str] = []
        if bit in data_bits:
            modes.append("data_only")
        if bit in model_bits:
            modes.append("model_only")
        if self.config.enable_hybrid and bit in data_bits and bit in model_bits:
            modes.append("hybrid")
        return modes

    def _skip_bit_for_dtype_profile(self, bit_width: int) -> bool:
        return self.config.dtype_profile == "float16" and bit_width != 16

    def _load_dataset(
        self,
        dataset_source: DatasetSourceType,
        dataset_name: SklearnDatasetName | CSVDatasetName,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if dataset_source == DatasetSourceType.SKLEARN:
            dataset = DatasetFactory.create(dataset_source, name=dataset_name, random_state=seed)
        elif dataset_source == DatasetSourceType.CSV:
            target_column = CSV_DATASET_TARGET_COLUMNS.get(dataset_name)
            if target_column is None:
                raise ValueError(f"Target column not defined for CSV dataset: {dataset_name}.")
            dataset = DatasetFactory.create(
                dataset_source,
                file_path=dataset_name,
                target_column=target_column,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unsupported dataset source type: {dataset_source}.")

        x_train, x_test, y_train, y_test = dataset.load_data()
        return (
            np.asarray(x_train, dtype=np.float64),
            np.asarray(x_test, dtype=np.float64),
            np.asarray(y_train),
            np.asarray(y_test),
        )

    @staticmethod
    def _build_model(model_class: type[BaseModel], model_params: dict[str, Any]) -> BaseModel:
        params = dict(model_params)
        try:
            return model_class(params=params)
        except TypeError:
            return model_class(**params)

    @staticmethod
    def _fit_model(model: BaseModel, x_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            model.train(x_train, y_train)
        except TypeError:
            model.train(x_train)

    def _build_execution_id(
        self,
        *,
        dataset_name: SklearnDatasetName | CSVDatasetName,
        algorithm_name: str,
        mode: str,
        bit_width: int,
        seed: int,
    ) -> str:
        parts = [
            self._dataset_label(dataset_name),
            algorithm_name,
            mode,
            str(bit_width),
            self.config.dtype_profile,
            self.config.method,
            str(seed),
        ]
        return "|".join(parts)

    def _build_metadata(self, mode: str, execution_id: str, bit_width: int) -> dict[str, Any]:
        config_payload = asdict(self.config)
        config_hash = hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return {
            "mode": mode,
            "execution_id": execution_id,
            "config_hash": config_hash,
            "target_bit_width": bit_width,
        }

    @staticmethod
    def _to_result_experiment_type(mode: str) -> str:
        mapping = {"data_only": "data_quant", "model_only": "model_quant", "hybrid": "hybrid"}
        return mapping[mode]

    @staticmethod
    def _dataset_label(dataset_name: SklearnDatasetName | CSVDatasetName) -> str:
        return dataset_name.value if hasattr(dataset_name, "value") else str(dataset_name)

    @staticmethod
    def _coalesce_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _estimate_model_memory_bytes(model: BaseModel) -> int:
        estimator = getattr(model, "model", model)
        arrays = QuantizationExperiment._extract_model_arrays(estimator)
        return int(sum(np.asarray(arr).nbytes for arr in arrays))

    @staticmethod
    def _estimate_quantized_model_memory_bytes(model: BaseModel, bit_width: int) -> int:
        estimator = getattr(model, "model", model)
        arrays = QuantizationExperiment._extract_model_arrays(estimator)
        bytes_per_value = max(bit_width // 8, 1)
        return int(sum(np.asarray(arr).size * bytes_per_value for arr in arrays))

    @staticmethod
    def _compute_model_quantization_mse(
        baseline_model: BaseModel, quantized_model: BaseModel
    ) -> float:
        base_est = getattr(baseline_model, "model", baseline_model)
        quant_est = getattr(quantized_model, "model", quantized_model)
        base_arrays = QuantizationExperiment._extract_model_arrays(base_est)
        quant_arrays = QuantizationExperiment._extract_model_arrays(quant_est)
        if not base_arrays or len(base_arrays) != len(quant_arrays):
            return 0.0

        errors: list[float] = []
        sizes: list[int] = []
        for base_arr, quant_arr in zip(base_arrays, quant_arrays):
            b_arr = np.asarray(base_arr, dtype=np.float64).reshape(-1)
            q_arr = np.asarray(quant_arr, dtype=np.float64).reshape(-1)
            if b_arr.shape != q_arr.shape:
                continue
            errors.append(float(np.mean((b_arr - q_arr) ** 2)))
            sizes.append(int(b_arr.size))
        if not errors or not sizes or sum(sizes) == 0:
            return 0.0
        return float(np.average(errors, weights=sizes))

    @staticmethod
    def _extract_model_arrays(estimator: Any) -> list[np.ndarray]:
        arrays: list[np.ndarray] = []

        if hasattr(estimator, "coefs_"):
            arrays.extend([np.asarray(v, dtype=np.float64) for v in estimator.coefs_])
        if hasattr(estimator, "intercepts_"):
            arrays.extend([np.asarray(v, dtype=np.float64) for v in estimator.intercepts_])

        if hasattr(estimator, "tree_") and hasattr(estimator.tree_, "threshold"):
            threshold = np.asarray(estimator.tree_.threshold, dtype=np.float64)
            valid = threshold[threshold != -2.0]
            if valid.size > 0:
                arrays.append(valid)

        fit_x = getattr(estimator, "_fit_X", None)
        if fit_x is not None:
            arrays.append(np.asarray(fit_x, dtype=np.float64))

        if hasattr(estimator, "cluster_centers_"):
            arrays.append(np.asarray(estimator.cluster_centers_, dtype=np.float64))
        if hasattr(estimator, "means_"):
            arrays.append(np.asarray(estimator.means_, dtype=np.float64))
        if hasattr(estimator, "covariances_"):
            arrays.append(np.asarray(estimator.covariances_, dtype=np.float64))
        return arrays
