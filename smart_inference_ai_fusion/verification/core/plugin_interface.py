"""Interface base para plugins de verificadores formais."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import time

from ...utils.logging import logger


class VerificationStatus(Enum):
    """Status do resultado da verificação."""
    SUCCESS = "success"
    FAILURE = "failure" 
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class VerificationResult:
    """Resultado de uma verificação formal."""
    status: VerificationStatus
    verifier_name: str
    execution_time: float
    message: str = ""
    constraints_checked: List[str] = None
    constraints_satisfied: List[str] = None
    constraints_violated: List[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.constraints_checked is None:
            self.constraints_checked = []
        if self.constraints_satisfied is None:
            self.constraints_satisfied = []
        if self.constraints_violated is None:
            self.constraints_violated = []
        if self.details is None:
            self.details = {}
    
    @property
    def success(self) -> bool:
        """True se a verificação foi bem-sucedida."""
        return self.status == VerificationStatus.SUCCESS


@dataclass 
class VerificationInput:
    """Entrada para verificação formal."""
    name: str  # Nome da transformação/função sendo verificada
    constraints: Dict[str, Any]  # Constraints a serem verificados
    input_data: Any = None
    output_data: Any = None
    parameters: Dict[str, Any] = None
    timeout: float = 30.0
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class FormalVerifier(ABC):
    """Interface base para verificadores formais."""
    
    def __init__(self, name: str):
        self.name = name
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        """Se o verificador está habilitado."""
        return self._enabled and self.is_available()
    
    def enable(self):
        """Habilita o verificador."""
        self._enabled = True
    
    def disable(self):
        """Desabilita o verificador."""
        self._enabled = False
    
    @abstractmethod
    def is_available(self) -> bool:
        """Verifica se o verificador está disponível (dependências instaladas)."""
        pass
    
    @abstractmethod
    def supported_constraints(self) -> List[str]:
        """Lista de tipos de constraints suportados pelo verificador."""
        pass
    
    @abstractmethod
    def verify(self, input_data: VerificationInput) -> VerificationResult:
        """Executa a verificação formal."""
        pass
    
    def can_verify(self, constraints: Dict[str, Any]) -> bool:
        """Verifica se pode processar os constraints dados."""
        if not self.enabled:
            return False
        
        supported = self.supported_constraints()
        constraint_keys = list(constraints.keys())
        can_verify_result = any(constraint_type in supported for constraint_type in constraint_keys)
        
        # DEBUG: Log para debugging
        if self.name == "Z3":
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Z3 can_verify DEBUG:")
            logger.info(f"  - Constraints recebidas: {constraint_keys}")
            logger.info(f"  - Constraints suportadas: {supported}")
            logger.info(f"  - Pode verificar: {can_verify_result}")
        
        return can_verify_result


class VerifierRegistry:
    """Registro de verificadores formais disponíveis."""
    
    def __init__(self):
        self._verifiers: Dict[str, FormalVerifier] = {}
    
    def register(self, verifier: FormalVerifier):
        """Registra um novo verificador."""
        self._verifiers[verifier.name] = verifier
        logger.info("✅ Registered verifier: %s", verifier.name)
    
    def get_verifier(self, name: str) -> Optional[FormalVerifier]:
        """Obtém um verificador pelo nome."""
        return self._verifiers.get(name)
    
    def get_available_verifiers(self) -> List[FormalVerifier]:
        """Lista verificadores disponíveis e habilitados."""
        return [v for v in self._verifiers.values() if v.enabled]
    
    def get_verifiers_for_constraints(self, constraints: Dict[str, Any]) -> List[FormalVerifier]:
        """Obtém verificadores capazes de processar os constraints dados."""
        return [v for v in self.get_available_verifiers() if v.can_verify(constraints)]
    
    def list_verifiers(self) -> Dict[str, Dict[str, Any]]:
        """Lista informações de todos os verificadores."""
        info = {}
        for name, verifier in self._verifiers.items():
            info[name] = {
                'available': verifier.is_available(),
                'enabled': verifier.enabled,
                'supported_constraints': verifier.supported_constraints()
            }
        return info


# Instância global do registro
registry = VerifierRegistry()
