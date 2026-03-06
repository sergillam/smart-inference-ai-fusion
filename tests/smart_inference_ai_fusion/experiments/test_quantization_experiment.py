"""Integration tests for QuantizationExperiment."""

from smart_inference_ai_fusion.experiments.quantization_experiment import QuantizationExperiment
from smart_inference_ai_fusion.models.knn_model import KNNModel
from smart_inference_ai_fusion.models.minibatch_kmeans_model import MiniBatchKMeansModel
from smart_inference_ai_fusion.quantization.core import QuantizationConfig
from smart_inference_ai_fusion.utils.types import DatasetSourceType, SklearnDatasetName


def test_run_supervised_returns_three_modes_for_single_bit() -> None:
    """With one bit width and hybrid enabled, three modes should be produced."""
    config = QuantizationConfig(data_bits=(16,), model_bits=(16,), method="uniform", random_seed=42)
    experiment = QuantizationExperiment(config)
    results = experiment.run_supervised(
        DatasetSourceType.SKLEARN,
        SklearnDatasetName.WINE,
        KNNModel,
        {"n_neighbors": 3},
        seed=42,
    )
    assert len(results) == 3
    modes = {result.metadata["mode"] for result in results}
    assert modes == {"data_only", "model_only", "hybrid"}
    assert all(result.baseline_accuracy is not None for result in results)


def test_run_supervised_honors_skip_execution_ids() -> None:
    """Previously executed IDs should be skipped deterministically."""
    config = QuantizationConfig(data_bits=(16,), model_bits=(16,), method="uniform", random_seed=42)
    experiment = QuantizationExperiment(config)
    first_run = experiment.run_supervised(
        DatasetSourceType.SKLEARN,
        SklearnDatasetName.WINE,
        KNNModel,
        {"n_neighbors": 3},
        seed=42,
    )
    executed = {result.metadata["execution_id"] for result in first_run}
    second_run = experiment.run_supervised(
        DatasetSourceType.SKLEARN,
        SklearnDatasetName.WINE,
        KNNModel,
        {"n_neighbors": 3},
        seed=42,
        skip_execution_ids=executed,
    )
    assert second_run == []


def test_run_unsupervised_returns_silhouette_metrics() -> None:
    """Unsupervised flow should return silhouette fields in quantization results."""
    config = QuantizationConfig(data_bits=(16,), model_bits=(16,), method="uniform", random_seed=42)
    experiment = QuantizationExperiment(config)
    results = experiment.run_unsupervised(
        DatasetSourceType.SKLEARN,
        SklearnDatasetName.MAKE_BLOBS,
        MiniBatchKMeansModel,
        {"n_clusters": 3, "random_state": 42},
        seed=42,
    )
    assert len(results) == 3
    assert all(result.baseline_silhouette is not None for result in results)
