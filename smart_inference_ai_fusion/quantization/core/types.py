"""Base type aliases used by SIP-Q quantization components."""

from typing import Literal

QuantMethod = Literal["uniform", "minmax", "kmeans", "percentile"]
BitWidth = Literal[8, 16, 32]
DTypeProfile = Literal["integer", "float16"]
