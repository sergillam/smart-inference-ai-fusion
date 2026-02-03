"""Sistema de Verificação Formal - Interface de Plugins."""

# Interface principal
from .core.formal_verification import (
    disable_verification,
    enable_verification,
    get_available_verifiers,
    list_verifiers,
    verification_manager,
    verify,
)

# Classes base para criação de plugins
from .core.plugin_interface import (
    FormalVerifier,
    VerificationInput,
    VerificationResult,
    VerificationStatus,
    registry,
)

__all__ = [
    # Interface de uso principal
    "verify",
    "enable_verification",
    "disable_verification",
    "list_verifiers",
    "get_available_verifiers",
    "verification_manager",
    # Para desenvolver novos plugins
    "FormalVerifier",
    "VerificationInput",
    "VerificationResult",
    "VerificationStatus",
    "registry",
]

__version__ = "2.0.0"
