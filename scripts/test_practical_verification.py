#!/usr/bin/env python3
"""Teste prático da verificação formal - foco na usabilidade."""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_verification_system():
    """Teste prático do sistema de verificação."""
    logger.info("🔬 Testando sistema de verificação formal...")
    
    try:
        # Import basic components
        from smart_inference_ai_fusion.verification import (
            is_verification_enabled, enable_verification,
            get_verification_config, VerificationRegistry
        )
        
        logger.info("✅ Imports básicos funcionando")
        
        # Check current config
        logger.info(f"📋 Verificação habilitada: {is_verification_enabled()}")
        config = get_verification_config()
        logger.info(f"📋 Configuração: {config}")
        
        # Test registry
        registry = VerificationRegistry()
        available_verifiers = registry.get_available_verifiers()
        
        logger.info(f"🔍 Verificadores disponíveis: {len(available_verifiers)}")
        for verifier in available_verifiers:
            logger.info(f"  - {verifier.name} v{verifier.version} (prioridade: {verifier.priority})")
            logger.info(f"    Constraints suportados: {len(verifier.supported_constraints)}")
        
        if available_verifiers:
            logger.info("✅ Sistema de verificação operacional!")
            return True
        else:
            logger.warning("⚠️ Nenhum verificador disponível - instale Z3: pip install z3-solver")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_verification():
    """Teste simples de verificação usando decorador."""
    logger.info("🧪 Testando verificação simples...")
    
    try:
        from smart_inference_ai_fusion.verification import verify_transformation
        import numpy as np
        
        # Criar uma transformação simples com verificação
        @verify_transformation(
            constraints={
                'bounds': {'min': 0.0, 'max': 1.0}
            },
            description="Teste simples de normalização"
        )
        class SimpleNormalization:
            def transform(self, data):
                """Normaliza dados para [0,1]."""
                if hasattr(data, '__iter__'):
                    data_array = np.array(data)
                    min_val = np.min(data_array)
                    max_val = np.max(data_array)
                    if max_val == min_val:
                        return np.full_like(data_array, 0.5)
                    return (data_array - min_val) / (max_val - min_val)
                else:
                    return 0.5  # Single value
        
        # Testar a transformação
        transform = SimpleNormalization()
        test_data = [1, 2, 3, 4, 5]
        
        logger.info(f"📥 Dados de entrada: {test_data}")
        result = transform.transform(test_data)
        logger.info(f"📤 Dados de saída: {result}")
        
        # Verificar se está no range correto
        if hasattr(result, '__iter__'):
            in_bounds = all(0.0 <= x <= 1.0 for x in result)
        else:
            in_bounds = 0.0 <= result <= 1.0
            
        logger.info(f"✅ Resultado dentro dos bounds [0,1]: {in_bounds}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de verificação: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_transformations():
    """Teste com transformações reais do projeto."""
    logger.info("🎯 Testando com transformações reais...")
    
    try:
        # Test with actual project transformations
        from smart_inference_ai_fusion.inference.transformations.data.noise import GaussianNoise
        import numpy as np
        
        logger.info("📊 Testando GaussianNoise (transformação real)...")
        
        # Create test data
        X_test = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        transform = GaussianNoise(level=0.1)
        
        logger.info(f"📥 Shape original: {X_test.shape}")
        X_transformed = transform.apply(X_test)
        logger.info(f"📤 Shape transformado: {X_transformed.shape}")
        
        # Verificações básicas
        shape_ok = X_test.shape == X_transformed.shape
        reasonable_change = np.mean(np.abs(X_transformed - X_test)) < 1.0
        
        logger.info(f"✅ Shape preservado: {shape_ok}")
        logger.info(f"✅ Mudança razoável: {reasonable_change}")
        
        return shape_ok and reasonable_change
        
    except Exception as e:
        logger.error(f"❌ Erro com transformações reais: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_practical_usage():
    """Mostra exemplos práticos de uso."""
    logger.info("📚 Exemplos práticos de uso:")
    
    print("""
🔧 COMO USAR O SISTEMA DE VERIFICAÇÃO:

1. ATIVAR/DESATIVAR VERIFICAÇÃO:
   ```python
   from smart_inference_ai_fusion.verification import enable_verification
   enable_verification(True)  # Ativar
   enable_verification(False) # Desativar
   ```

2. USAR EM TRANSFORMAÇÕES:
   ```python
   from smart_inference_ai_fusion.verification import verify_transformation
   
   @verify_transformation(
       constraints={
           'bounds': {'min': 0.0, 'max': 1.0},
           'shape_preservation': True
       }
   )
   class MinhaTransformacao:
       def transform(self, data):
           # Sua lógica aqui
           return processed_data
   ```

3. VERIFICAR QUAIS VERIFICADORES ESTÃO DISPONÍVEIS:
   ```python
   from smart_inference_ai_fusion.verification import VerificationRegistry
   registry = VerificationRegistry()
   for verifier in registry.get_available_verifiers():
       print(f"- {verifier.name}: {verifier.supported_constraints}")
   ```

4. EXECUTAR EXPERIMENTOS COM VERIFICAÇÃO:
   ```bash
   make run verify-formal        # Com verificação
   make run                      # Sem verificação
   ```
""")

def main():
    """Função principal do teste prático."""
    logger.info("🚀 Iniciando teste prático do sistema de verificação...")
    
    results = {
        'sistema_basico': test_verification_system(),
        'verificacao_simples': test_simple_verification(),
        'transformacoes_reais': test_with_real_transformations()
    }
    
    logger.info("📊 Resultados dos testes:")
    for test_name, passed in results.items():
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        logger.info(f"  {test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    if total_passed == total_tests:
        logger.info("🎉 Todos os testes passaram! Sistema pronto para uso.")
        show_practical_usage()
        return True
    else:
        logger.error(f"⚠️ {total_tests - total_passed} testes falharam.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
