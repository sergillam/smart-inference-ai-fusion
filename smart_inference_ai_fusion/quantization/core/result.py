"""Result schema for SIP-Q quantization experiments."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from smart_inference_ai_fusion.quantization.core.types import BitWidth, DTypeProfile, QuantMethod


class QuantizationResult(BaseModel):
    """Standardized output payload for quantization runs."""

    experiment_type: Literal["data_quant", "model_quant", "hybrid"]
    dataset_name: str
    algorithm_name: str
    quantization_method: QuantMethod
    bit_width: BitWidth
    dtype_profile: DTypeProfile = "integer"

    baseline_accuracy: Optional[float] = None
    quantized_accuracy: Optional[float] = None
    accuracy_degradation: Optional[float] = None

    baseline_silhouette: Optional[float] = None
    quantized_silhouette: Optional[float] = None
    silhouette_degradation: Optional[float] = None

    baseline_memory_bytes: int = Field(gt=0)
    quantized_memory_bytes: int = Field(gt=0)
    compression_ratio: float = Field(gt=0)

    baseline_time_ms: float = Field(gt=0)
    quantized_time_ms: float = Field(gt=0)
    overhead_pct: float = Field(ge=-100)

    quantization_mse: float = Field(ge=0)
    seed: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_consistency(self) -> "QuantizationResult":
        """Enforce basic cross-field consistency constraints."""
        if self.quantized_memory_bytes > self.baseline_memory_bytes:
            raise ValueError("quantized_memory_bytes cannot exceed baseline_memory_bytes.")
        return self
