"""Public API for SIP-Q quantization module."""

from smart_inference_ai_fusion.quantization.core.config import QuantizationConfig
from smart_inference_ai_fusion.quantization.core.methods import (
    dequantize,
    kmeans_quantize,
    minmax_quantize,
    percentile_quantize,
    uniform_quantize,
)
from smart_inference_ai_fusion.quantization.core.result import QuantizationResult
from smart_inference_ai_fusion.quantization.core.types import BitWidth, DTypeProfile, QuantMethod
from smart_inference_ai_fusion.quantization.data.feature_quantizer import FeatureQuantizer
from smart_inference_ai_fusion.quantization.evaluation.benchmark import (
    benchmark_inference,
    compute_compression_ratio,
    compute_memory_reduction,
    compute_overhead_pct,
    estimate_memory_bytes,
)
from smart_inference_ai_fusion.quantization.evaluation.metrics import (
    compute_clustering_metrics,
    compute_quantization_mse,
    compute_supervised_metrics,
)
from smart_inference_ai_fusion.quantization.model.weight_quantizer import WeightQuantizer

__all__ = [
    "BitWidth",
    "DTypeProfile",
    "QuantMethod",
    "QuantizationConfig",
    "QuantizationResult",
    "uniform_quantize",
    "minmax_quantize",
    "kmeans_quantize",
    "percentile_quantize",
    "dequantize",
    "FeatureQuantizer",
    "WeightQuantizer",
    "compute_supervised_metrics",
    "compute_clustering_metrics",
    "compute_quantization_mse",
    "benchmark_inference",
    "compute_memory_reduction",
    "compute_compression_ratio",
    "estimate_memory_bytes",
    "compute_overhead_pct",
]
