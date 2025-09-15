"""Interface principal para verificação formal."""

from typing import Any, Dict, List, Optional
import logging

from .plugin_interface import (
    VerificationInput, VerificationResult, VerificationStatus, 
    FormalVerifier, registry
)

logger = logging.getLogger(__name__)


class FormalVerificationManager:
    """Manager principal para verificação formal."""
    
    def __init__(self):
        self.enabled = True
        # Carregar todos os plugins disponíveis
        self._load_plugins()
    
    def _load_plugins(self):
        """Carrega todos os plugins de verificação disponíveis."""
        try:
            # Importar plugin Z3
            from ..plugins import z3_plugin
            logger.debug("Z3 plugin loaded")
        except ImportError as e:
            logger.warning(f"Could not load Z3 plugin: {e}")
        
        # Aqui você pode adicionar imports de outros plugins no futuro
        # from ..plugins import bmc_plugin
        # from ..plugins import cbmc_plugin
        # etc.
    
    def enable_verification(self):
        """Habilita verificação formal globalmente."""
        self.enabled = True
        logger.info("Formal verification enabled")
    
    def disable_verification(self):
        """Desabilita verificação formal globalmente.""" 
        self.enabled = False
        logger.info("Formal verification disabled")
    
    def verify(self, name: str, constraints: Dict[str, Any], 
               input_data: Any = None, output_data: Any = None,
               parameters: Dict[str, Any] = None, timeout: float = 30.0,
               verifier_name: Optional[str] = None) -> VerificationResult:
        """
        Executa verificação formal.
        
        Args:
            name: Nome da transformação/função sendo verificada
            constraints: Dicionário de constraints a verificar
            input_data: Dados de entrada (opcional)
            output_data: Dados de saída (opcional) 
            parameters: Parâmetros adicionais (opcional)
            timeout: Timeout em segundos
            verifier_name: Nome específico do verificador (None = auto-select)
        
        Returns:
            Resultado da verificação
        """
        if not self.enabled:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                verifier_name="Manager",
                execution_time=0.0,
                message="Formal verification is disabled"
            )
        
        # Criar input de verificação
        verification_input = VerificationInput(
            name=name,
            constraints=constraints,
            input_data=input_data,
            output_data=output_data,
            parameters=parameters or {},
            timeout=timeout
        )
        
        # Selecionar verificador
        if verifier_name:
            verifier = registry.get_verifier(verifier_name)
            if not verifier:
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    verifier_name=verifier_name,
                    execution_time=0.0,
                    message=f"Verifier '{verifier_name}' not found"
                )
        else:
            # Auto-seleção: pegar o primeiro verificador que suporta os constraints
            verifiers = registry.get_verifiers_for_constraints(constraints)
            if not verifiers:
                return VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    verifier_name="Auto-select",
                    execution_time=0.0,
                    message="No verifiers available for given constraints"
                )
            verifier = verifiers[0]  # Pegar o primeiro disponível
        
        # Executar verificação
        logger.info(f"Running verification '{name}' with {verifier.name}")
        return verifier.verify(verification_input)
    
    def list_verifiers(self) -> Dict[str, Dict[str, Any]]:
        """Lista todos os verificadores registrados."""
        return registry.list_verifiers()
    
    def get_available_verifiers(self) -> List[str]:
        """Obtém nomes dos verificadores disponíveis."""
        return [v.name for v in registry.get_available_verifiers()]
    
    def enable_verifier(self, name: str) -> bool:
        """Habilita um verificador específico."""
        verifier = registry.get_verifier(name)
        if verifier:
            verifier.enable()
            logger.info(f"Enabled verifier: {name}")
            return True
        return False
    
    def disable_verifier(self, name: str) -> bool:
        """Desabilita um verificador específico."""
        verifier = registry.get_verifier(name)
        if verifier:
            verifier.disable()
            logger.info(f"Disabled verifier: {name}")
            return True
        return False


# Instância global do manager
verification_manager = FormalVerificationManager()

# Funções de conveniência para uso direto
def verify(name: str, constraints: Dict[str, Any], **kwargs) -> VerificationResult:
    """Função de conveniência para verificação formal."""
    return verification_manager.verify(name, constraints, **kwargs)

def enable_verification():
    """Habilita verificação formal."""
    verification_manager.enable_verification()

def disable_verification():
    """Desabilita verificação formal."""
    verification_manager.disable_verification()

def list_verifiers() -> Dict[str, Dict[str, Any]]:
    """Lista verificadores disponíveis."""
    return verification_manager.list_verifiers()

def get_available_verifiers() -> List[str]:
    """Obtém verificadores disponíveis."""
    return verification_manager.get_available_verifiers()
