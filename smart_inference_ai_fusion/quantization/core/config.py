"""Configuration schema for SIP-Q quantization experiments."""

from dataclasses import dataclass
from warnings import warn

from smart_inference_ai_fusion.quantization.core.types import BitWidth, DTypeProfile, QuantMethod

_VALID_BITS = {8, 16, 32}
_VALID_METHODS = {"uniform", "minmax", "kmeans", "percentile"}
_VALID_DTYPE_PROFILES = {"integer", "float16"}


@dataclass(frozen=True)
class QuantizationConfig:
    """Immutable configuration for SIP-Q experiments."""

    data_bits: tuple[BitWidth, ...] = (8, 16, 32)
    model_bits: tuple[BitWidth, ...] = (8, 16, 32)
    dtype_profile: DTypeProfile = "integer"
    method: QuantMethod = "uniform"
    enable_hybrid: bool = True
    calibration_samples: int = 1000
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate semantic invariants beyond static typing."""
        self._validate_bits(self.data_bits, "data_bits")
        self._validate_bits(self.model_bits, "model_bits")
        self._validate_unique_bits(self.data_bits, "data_bits")
        self._validate_unique_bits(self.model_bits, "model_bits")

        if self.dtype_profile not in _VALID_DTYPE_PROFILES:
            raise ValueError(f"dtype_profile must be one of {_VALID_DTYPE_PROFILES}.")

        if self.method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}.")

        if self.calibration_samples <= 0:
            raise ValueError("calibration_samples must be > 0.")

        if self.random_seed < 0:
            raise ValueError("random_seed must be >= 0.")

        if self.dtype_profile == "float16" and self.method != "uniform":
            warn(
                "float16 usa cast direto - o campo 'method' e ignorado neste perfil.",
                UserWarning,
                stacklevel=2,
            )

    @staticmethod
    def _validate_bits(bits: tuple[BitWidth, ...], field_name: str) -> None:
        if not bits:
            raise ValueError(f"{field_name} cannot be empty.")
        invalid_values = [value for value in bits if value not in _VALID_BITS]
        if invalid_values:
            raise ValueError(
                f"{field_name} contains unsupported bit widths: {invalid_values}. "
                f"Expected values from {_VALID_BITS}."
            )

    @staticmethod
    def _validate_unique_bits(bits: tuple[BitWidth, ...], field_name: str) -> None:
        if len(set(bits)) != len(bits):
            raise ValueError(f"{field_name} cannot contain duplicate bit widths.")
